from collections.abc import Callable
from pathlib import Path
import torch

from dltool.utils import cpu_state_dict


def watch_best_state_hook(metric: str, select_fn: Callable = min):
    def hook(trainer):
        if (history := trainer.logger.history) and (metric_values := history[metric]):
            if select_fn(metric_values) == metric_values[-1]:
                trainer.best_model_state = cpu_state_dict(trainer.algorithm)
    return hook


def save_state_hook(path: str | Path, best: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def hook(trainer):
        state = trainer.best_model_state if best else cpu_state_dict(trainer.algorithm)
        if state is not None:
            torch.save(state, path)
    return hook