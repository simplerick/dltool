from collections import defaultdict
import numpy as np
from collections.abc import Sequence, Mapping, Callable
import torch
import logging
import re
import yaml


class ConsoleLogger(logging.Logger):
    color_bold = "\033[1m"
    color_darkcyan = "\033[36m"
    color_darkblue = "\033[34m"
    color_end = "\033[0m"

    def __init__(self, name, formatting: str = None):
        super().__init__(name)
        formatting = formatting or \
            f"{ConsoleLogger.color_bold}{ConsoleLogger.color_darkblue}%(name)s{ConsoleLogger.color_end}: %(message)s"
        formatter = logging.Formatter(
            formatting,
            datefmt="%d-%b-%y %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)

def unflatten_dict(d: dict, sep: str = "."):
    """
    Unflatten a dictionary with keys like "a.b.c" to a nested dictionary. Supports regex separators.
    """
    unflatten_d = {}
    for k, v in d.items():
        k_lvls = re.split(sep, k)
        cur_d = unflatten_d
        for l in k_lvls[:-1]:
            if l not in cur_d:
                cur_d[l] = {}
            cur_d = cur_d[l]
        cur_d[k_lvls[-1]] = v
    return unflatten_d

def format_metrics_dict_to_str(d: dict, step: int = None) -> str:
    """
    Formats a dictionary to a beautiful string.
    """
    grouped_d = {k if "/" in k else "/"+k: v for k, v in d.items()}
    grouped_d = {k: f"{v:.4g}" for k, v in grouped_d.items()}
    grouped_d = unflatten_dict(grouped_d, "[/.]")
    s = [f"step {step}\n" if step is not None else "\n"]
    for group, metrics in grouped_d.items():
        s.append(f"{ConsoleLogger.color_darkcyan}{group}{ConsoleLogger.color_end}")
        s.append(yaml.dump({' ': metrics} if group else metrics).replace("\'", ""))
    return "".join(s)


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


class NotLogger:
    """
    Dummy logger
    """
    def __getattribute__(self, item):
        if item in ["log", "store", "clear", "flush", "plot"]:
            return lambda *args, **kwargs: None


class Logger:
    """
    Logger, supports the wandb library as backend.
    """
    def __new__(cls, api, *args, **kwargs):
        if api is None:
            return super().__new__(NotLogger)
        for m in ["log", "run", "log_artifact", "Artifact"]:
            if not hasattr(api, m):
                raise ValueError(f"Api doesn't have {m} method. Currently only wandb is supported.")
        return super().__new__(cls)

    def __init__(self,
                 wandb_api,
                 logging_freq: int = 1,
                 console_output: bool = True,
                 scalar_aggregation: Callable = np.mean,
                 tensor_aggregation: Callable = torch.cat):
        self.api = wandb_api
        self.console_logger = ConsoleLogger(__name__) if console_output else None
        self.logging_freq = logging_freq
        self.scalar_aggregation = scalar_aggregation
        self.tensor_aggregation = tensor_aggregation
        self._step = 0  # last step
        self._max_step = 0  # maximum step
        self._last_writing_step = {}  # steps of last writing for each group
        self._data_storage: dict = defaultdict(list)  # {g: [{m: v, }, {m: v, }, ], }
        self._data_types: dict = defaultdict(dict)  # {g: {m: t, }, }
        self.history = defaultdict(list)

    def log(self, metrics, step: int, group: str = None, accumulate: bool = False, write: bool = False):
        next_log_step = ((self._max_step // self.logging_freq) + 1) * self.logging_freq - 1
        if group in self._last_writing_step and step <= self._last_writing_step[group]:
            raise ValueError(f"step must be increasing: {step} <= {self._last_writing_step[group]}")
        if step > next_log_step and not accumulate:
            self.write(next_log_step)

        self.store(metrics, group)
        self._step = step
        self._max_step = max(self._max_step, step)
        if write:
            self.write(step, [group])

    def store(self, metrics, group: str = None):
        if isinstance(metrics, dict):
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
            raise ValueError

    def clear(self):
        self._data_storage = defaultdict(list)
        self._data_types = defaultdict(dict)

    @staticmethod
    def path_cat(*parts: str):
        return "/".join([x for x in parts if x])

    def write(self, step: int, groups: Sequence[str] = None, metric_names: Sequence[str] = None):
        scalar_metrics, tensor_metrics = {}, {}
        for g in self._data_storage.copy():
            if groups is not None and g not in groups:
                continue
            self._last_writing_step[g] = step
            data = self._data_storage.pop(g)
            g_str = str(g) if g is not None else ""
            metrics_to_write = metric_names or self._data_types[g].keys()
            # average scalar metrics
            scalar_metric_names = [m for m in metrics_to_write if self._data_types[g][m] in [float, int]]
            scalar_metrics.update({Logger.path_cat(g_str, n) : m for n, m in
                summarize(data, self.scalar_aggregation, names=scalar_metric_names, pop=True).items()})
            # stack tensor metrics
            tensor_metric_names = [m for m in metrics_to_write if self._data_types[g][m] == torch.Tensor]
            if tensor_metric_names:
                tensor_metrics[g_str] = {n: m for n, m in
                    summarize(data, self.tensor_aggregation, names=tensor_metric_names, pop=True).items()}
        # log scalar metrics
        if scalar_metrics:
            self.api.log(scalar_metrics, step)
            if self.console_logger:
                self.console_logger.info(format_metrics_dict_to_str(scalar_metrics, step))
            for n, v in scalar_metrics.items():
                self.history[n].append(v)
        # save tensor metrics as artifact (dict with torch.tensors as wandb.Artifact)
        if tensor_metrics:
            artifact = self.api.Artifact(f"{self.api.run.id}_{step}", type='tensor')
            for g_str, metrics in tensor_metrics.items():
                with artifact.new_file(Logger.path_cat(g_str, "metrics.pt"), mode='wb') as f:
                    torch.save(metrics, f)
                self.history[g_str].append(metrics)
            self.api.log_artifact(artifact)

    def plot(self, x_key, y_key, group=None, stroke=None, title=None):
        # needs to be refactored!!!
        y, x = conditional_expectation(self._data_storage[group], condition_on=x_key)[y_key]
        data = list(zip(x, y))
        y_key = f"{group}/{y_key}" if group is not None else y_key
        table = self.api.Table(data=data, columns=[x_key, y_key])
        self.api.log({f"{y_key}:{x_key}": self.api.plot.line(table, x_key, y_key,
                                                             stroke=stroke, title=title)}, self._last_writing_step)
