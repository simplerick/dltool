from typing import Callable
import torch
from contextlib import contextmanager


class FuncModule(torch.nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.forward = func

    def __repr__(self):
        return f"FuncModule({self.forward.__name__})"


def apply(obj: object, func_str: str, **args) -> object:
    """
    Applies the specified method to the object;
    if there is no such method, it recursively traverses the object and tries to apply the method to its child objects.

    Args:
        obj: object
        func_str: method name
        **args: arguments to be passed to the method

    Returns:
        Resulting object, note that it is not necessarily a copy of the original object
    """
    if hasattr(obj, func_str):
        obj = getattr(obj, func_str)(**args)
    else:
        if isinstance(obj, list):
            obj[:] = [apply(x, func_str, **args) for x in obj]
        if isinstance(obj, dict):
            for k in obj:
                obj[k] = apply(obj[k], func_str, **args)
    return obj


def to(obj: object, **args) -> object:
    """
    Applies the "to" method to the object;
    if there is no such method, it recursively bypasses the object and tries to apply the method to its child objects.

    Args:
        obj: object
        **args: arguments to be passed to the "to" method

    Returns:
        Resulting object, note that it is not necessarily a copy of the original object
    """
    return apply(obj, "to", **args)


def detach(obj: object) -> object:
    """
    Applies the "detach" method to the object;
    if there is no such method, it recursively bypasses the object and tries to apply the method to its child objects.

    Args:
        obj: object

    Returns:
        Resulting object, note that it is not necessarily a copy of the original object
    """
    return apply(obj, "detach")


def cpu_state_dict(model: torch.nn.Module) -> dict:
    """
    Returns the model state dict in CPU memory.

    Args:
        model: torch module

    Returns:
        Dictionary containing a whole state of the module.
    """
    return to(model.state_dict(), device='cpu')


# context manager for evaluating
@contextmanager
def evaluating(model: torch.nn.Module):
    """
    Context-manager that temporarily switches the model to evaluation mode. Restores the initial training state of each submodule on exit.

    Args:
        model: torch module
    """
    with torch.no_grad():
        # save initial training state
        train_state = dict((m, m.training) for m in model.modules())
        try:
            model.eval()
            yield model
        finally:
            # restore initial training state
            for k, v in train_state.items():
                k.training = v
