from collections.abc import Callable
from pathlib import Path
import torch

from dltool.utils import cpu_state_dict


def watch_best_state_hook(metric: str, select_fn: Callable = min):
    def hook(trainer):
        metric_values = trainer.logger.history[metric]
        if len(metric_values) > 0 and select_fn(metric_values) == metric_values[-1]:
            trainer.best_model_state = cpu_state_dict(trainer.algorithm)
    return hook


def save_state_hook(path: str | Path, best: bool = True):
    def hook(trainer):
        state = trainer.best_model_state if best else cpu_state_dict(trainer.algorithm)
        if state is not None:
            torch.save(state, path)
    return hook