from collections import defaultdict
import numpy as np


def average(data: [], condition_on: str = None) -> dict:
    averaged = defaultdict(list)
    if condition_on is None:
        for records in data:
            for n, value in records.items():
                averaged[n].append(value)
        averaged = {n: np.mean(values) for n, values in averaged.items()}
    else:
        for records in data:
            for n, value in records.items():
                averaged[n].append([value, records.get(condition_on, np.inf)])
        for n, a in averaged.items():
            a = np.array(a).T
            unique = np.unique(a[1])
            m = (a[1] != unique.reshape(-1, 1))
            masked = np.ma.masked_array(np.broadcast_to(a[0], m.shape), m)
            averaged[n] = [masked.mean(1).data, unique]  # y,x
    return averaged


class Logger:
    """
    Logger, supports the wandb library.
    """

    def __init__(self, wandb, log_freq=1):
        self.api = wandb
        self.log_freq = log_freq
        self._log_step = 0  # data will be written with this step value
        self._step = 0  # last step
        self._data_storage: dict = defaultdict(list)  # {group : [{metric_name: value}, {metric_name: value}]}

    def log(self, metrics, step: int, group: str = None):
        self._step = step
        ix = step % self.log_freq
        if step < self._log_step:
            raise ValueError("incorrect step value")
        if step >= (self._log_step + self.log_freq):
            self.flush()
            self._log_step = step - ix
        self.store(metrics, group)

    def store(self, metrics, group: str = None):
        if isinstance(metrics, dict):
            self._data_storage[group].append(metrics)
        if isinstance(metrics, list):
            self._data_storage[group].extend(metrics)

    def clear(self):
        self._data_storage = defaultdict(list)

    def flush(self, metrics_to_write=None):
        for g, data in self._data_storage.items():
            g_str = f"{g}/" if g is not None else ""
            averaged_metrics = {g_str + n: m for n, m in average(data).items() if
                                (metrics_to_write is None or m in metrics_to_write)}
            self.api.log(averaged_metrics, self._log_step)
        self.clear()

    def plot(self, x_key, y_key, group=None, stroke=None, title=None):
        y, x = average(self._data_storage[group], condition_on=x_key)[y_key]
        data = list(zip(x, y))
        y_key = f"{group}/{y_key}" if group is not None else y_key
        table = self.api.Table(data=data, columns=[x_key, y_key])
        self.api.log({f"{y_key}:{x_key}": self.api.plot.line(table, x_key, y_key,
                                                             stroke=stroke, title=title)})
