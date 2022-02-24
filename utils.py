import torch
from contextlib import contextmanager


def to(obj, **args):
    if hasattr(obj, 'to'):
        obj = obj.to(**args)
    else:
        if isinstance(obj, list):
            obj[:] = [to(x, **args) for x in obj]
        if isinstance(obj, dict):
            for k in obj:
                obj[k] = to(obj[k], **args)
    return obj


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
