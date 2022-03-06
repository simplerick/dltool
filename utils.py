import torch
from contextlib import contextmanager


def apply(obj, func, **args):
    if hasattr(obj, func):
        obj = getattr(obj, func)(**args)
    else:
        if isinstance(obj, list):
            obj[:] = [apply(x, func, **args) for x in obj]
        if isinstance(obj, dict):
            for k in obj:
                obj[k] = apply(obj[k], func, **args)
    return obj


def to(obj, **args):
    return apply(obj, "to", **args)


def detach(obj):
    return apply(obj, "detach")


# context manager for evaluating
@contextmanager
def evaluating(model):
    """Temporarily switch to evaluation mode. Keeps original training state of every submodule"""
    with torch.no_grad():
        train_state = dict((m, m.training) for m in model.modules())
        try:
            model.eval()
            yield model
        finally:
            # restore initial training state
            for k, v in train_state.items():
                k.training = v
