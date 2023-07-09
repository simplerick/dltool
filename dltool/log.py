import warnings
from collections import defaultdict
from typing import Protocol
import numpy as np
from collections.abc import Sequence, Mapping, Callable
import torch
import logging
import yaml

from dltool.utils import flatten_dict


# ================================ Utils =================================


def format_dict_to_str(d: dict) -> str:
    """
    Formats a nested dictionary to a beautiful string.
    """
    def format_dict(x):
        if isinstance(x, dict):
           return {k: format_dict(v) for k, v in x.items()}
        if isinstance(x, (int, float)):
            return f"{x:.4g}"
    s = yaml.dump(format_dict(d))
    s = s.replace("\"", "").replace("\'", "").strip()
    return s


def summarize(data: Sequence[dict], function: Callable, names: Sequence[str] = None, pop: bool = False) -> dict:
    """
    Summarizes values for each key in names.

    Args:
        data: sequence of dictionaries
        function: summarization function
        names: for which variables to compute the conditional expectation
        pop: whether to remove the variables from records
    Returns:
        Dictionary with averaged values.
    """
    summarized = defaultdict(list)
    for records in data:
        for n in records.copy():
            if names is None or n in names:
                value = records.pop(n) if pop else records.get(n)
                summarized[n].append(value)
    averaged = {n: function(values) for n, values in summarized.items()}
    return averaged


def conditional_expectation(data: Sequence[Mapping], condition_on: str, names: Sequence[str] = None) -> dict:
    """
    Computes the conditional expectation for each key in names.

    Args:
        data: sequence of dictionaries
        names: for which variables to compute the conditional expectation.
        condition_on: for which variable to take the conditional expectation.
    Returns:
        Dictionary where values are pairs of numpy arrays.
    """
    averaged = defaultdict(list)
    for records in data:
        for n, value in records.items():
            if names is None or n in names:
                averaged[n].append([value, records.get(condition_on, np.inf)])
    for n, a in averaged.items():
        a = np.array(a).T
        unique = np.unique(a[1])
        m = (a[1] != unique.reshape(-1, 1))
        masked = np.ma.masked_array(np.broadcast_to(a[0], m.shape), m)
        averaged[n] = [masked.mean(1).data, unique]  # y,x
    return averaged


# ================================ Logger ================================

class TextLogger(logging.Logger):
    color_bold = "\033[1m"
    color_darkcyan = "\033[36m"
    color_darkblue = "\033[34m"
    color_end = "\033[0m"

    def __init__(self, name, formatting: str = None):
        super().__init__(name)
        formatting = formatting or \
            f"{TextLogger.color_bold}{TextLogger.color_darkblue}%(name)s{TextLogger.color_end}: %(message)s"
        formatter = logging.Formatter(
            formatting,
            datefmt="%d-%b-%y %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)


class WandBLike(Protocol):
    """Backend interface"""
    log: Callable
    run: object
    log_artifact: Callable
    Artifact: Callable


class History(defaultdict):
    """In-memory storage with structure {int: {str: num | tensor | dict}}"""
    def __init__(self):
        super().__init__(dict)

    def __getitem__(self, key: int | str) -> dict | list:
        """filter by int index or str attribute"""
        if isinstance(key, int):
            return super().__getitem__(key)
        if isinstance(key, str):
            if len(gm := key.split("/")) > 1:
                return [x[g][m] for x in super().values() if (g := gm[0]) in x and (m := gm[1]) in x[g]]
            return [x[key] for x in super().values() if key in x]

    def __contains__(self, item):
        raise NotImplementedError

    def get(self, item):
        raise NotImplementedError


class Logger:
    """
    Logger, supports the wandb library as a backend, in-memory storage and console logging.
    Accumulates values in memory and writes the aggregated values once in a specified number of steps.
    Supports metric grouping and custom aggregation functions.
    """
    def __init__(self,
                 backend: WandBLike = None,
                 console: bool = True,
                 in_memory: bool = True,
                 logging_freq: int = 1,
                 scalar_aggregation: Callable = np.mean,
                 tensor_aggregation: Callable = torch.cat):
        self.backend = backend
        self.console = TextLogger(__name__) if console else None
        self.history = History() if in_memory else None  # {step: {m: v, g: {m: v}}} - for storing written values in_memory
        self.logging_freq = logging_freq
        self.scalar_aggregation = scalar_aggregation
        self.tensor_aggregation = tensor_aggregation
        self.str_format = format_dict_to_str  # formatting function for text logging
        self._step = 0  # last step
        self._max_step = 0  # maximum step
        self._last_writing_step: dict = defaultdict(dict)  # {g: {m: step}} - for respecting chronology
        self._data_storage: dict = defaultdict(list)  # {g: [{m: v}]} - for accumulating before writing
        self._data_types: dict = defaultdict(dict)  # {g: {m: t}} - for checking data types
        self._check_backend_compatibility()
        self._check_aggregation_function()

    def _check_backend_compatibility(self):
        if self.backend is not None:
            if methods := [m for m in WandBLike.__annotations__ if not hasattr(self.backend, m)]:
                raise ValueError(f"backend must implement the following methods: {WandBLike.__annotations__}, "
                                 f"the passed backend does not implement {methods}.")

    def _check_aggregation_function(self):
        if not isinstance(t := self.scalar_aggregation([0]), (int, float)):
            warnings.warn(f"scalar aggregation function must return a number, got {type(t)}.")

    @staticmethod
    def _validate_name(name: str):
        not_allowed_symbols = ["\t", "\n", "\r", "\f", "\v", "\\", "/"]
        if any([s in name for s in not_allowed_symbols]):
            raise ValueError(f"metric or group name must not contain any of the following symbols: {not_allowed_symbols}.")

    def log(self, metrics, step: int, group: str = None, accumulate: bool = False, write: bool = False):
        """
        Collects metrics over a window of logging frequency length. The write operation is performed if:
            1) new log step exceeds the current window and accumulate is false (previous window will be committed)
            2) write is true (current window will be committed, the operation is group specific)
        Doesn't allow log step of a metric to be ≤ than the step of the previous write operation for that metric.
        """
        next_log_step = ((self._max_step // self.logging_freq) + 1) * self.logging_freq - 1
        if last_writing_steps := self._last_writing_step.get(group):
            if overwriting := {m: x for m in metrics if (x := last_writing_steps.get(m)) and step <= x}:
                raise ValueError(f"step must be increasing, last writing steps: {overwriting}, got step: {step}")
        if step > next_log_step and not accumulate:
            self.write(next_log_step)
        self.store(metrics, group)
        self._step = step
        self._max_step = max(self._max_step, step)
        if write:
            self.write(step, [group])

    def store(self, metrics, group: str = None):
        """
        Stores metrics in memory for a future write operation. Tracks data types and chronology.
        """
        if isinstance(metrics, dict):
            # validate names
            for m in metrics:
                self._validate_name(m)
            if group is None:
                if conflicts := [m for m in metrics if m in self._data_types]:
                    raise ValueError(f"metric name `{conflicts[0]}` is already being used by a group.")
            else:
                self._validate_name(group)
                # check that group name is not used by a metrics without group
                if any([m == group for m in self._data_types[None]]):
                    raise ValueError(f"group name `{group}` is already being used by a metric.")
            # update data types
            for m, v in metrics.items():
                t = self._data_types[group].get(m, type(v))
                self._data_types[group][m] = t if type(v) == t else object
                # detach if metric value is a tensor
                if isinstance(v, torch.Tensor):
                    metrics[m] = v.detach().cpu()
            # store metrics
            self._data_storage[group].append(metrics)
        else:
            raise ValueError(f"metrics must be a dict, got {type(metrics)}")

    @staticmethod
    def path_cat(*parts: str):
        return "/".join([x for x in parts if x])

    def write(self, step: int, groups: Sequence[str] = None, metrics: Sequence[str] = None):
        """
        Performs the write operation for the specified groups and metrics. Aggregates accumulated values
        and clears the storage for the write groups.
        """
        scalar_metrics, tensor_metrics = defaultdict(dict), defaultdict(dict)  #  {g: {m: v}}, {g: {m: v}}
        for g in groups or self._data_storage.copy():
            data = self._data_storage.pop(g)
            g_str = str(g) if g is not None else None
            metrics_to_write = metrics or self._data_types[g].keys()
            for m in metrics_to_write:
                self._last_writing_step[g][m] = step
                values = [x[m] for x in data if m in x]
                if values:
                    if self._data_types[g][m] in [float, int]:
                        scalar_metrics[g_str][m] = self.scalar_aggregation(values)
                    elif self._data_types[g][m] == torch.Tensor:
                        tensor_metrics[g_str][m] = self.tensor_aggregation(values)
        # log scalar metrics
        if scalar_metrics:
            scalar_metrics.update(scalar_metrics.pop(None, {}))  # disband 'non-group'
            if self.history is not None:
                self.history[step].update(scalar_metrics)
            if self.console is not None:
                self.console.info(f"step {step}\n" + self.str_format(scalar_metrics) + "\n")
            if self.backend is not None:
                self.backend.log(flatten_dict(scalar_metrics, "/"), step)
        # save tensor metrics
        if tensor_metrics:
            tensor_metrics.update(tensor_metrics.pop(None, {}))  # disband 'non-group'
            if self.history is not None:
                self.history[step].update(tensor_metrics)
            if self.backend is not None:  # as artifact (dict with torch.tensors as wandb.Artifact)
                artifact = self.backend.Artifact(f"{self.backend.run.id}", type='tensor')
                with artifact.new_file("metrics.pt", mode='wb') as f:
                    torch.save(tensor_metrics, f)
                self.backend.log_artifact(artifact, aliases=[str(step)])

    # def plot(self, x_key, y_key, group=None, stroke=None, title=None):
    #     # needs to be refactored!!!
    #     y, x = conditional_expectation(self._data_storage[group], condition_on=x_key)[y_key]
    #     data = list(zip(x, y))
    #     y_key = f"{group}/{y_key}" if group is not None else y_key
    #     table = self.api.Table(data=data, columns=[x_key, y_key])
    #     self.api.log({f"{y_key}:{x_key}": self.api.plot.line(table, x_key, y_key,
    #                                                          stroke=stroke, title=title)}, self._last_writing_step)
