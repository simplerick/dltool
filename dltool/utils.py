from collections.abc import Callable
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


def cartesian_prod(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Cartesian product of tensors. The same as torch.cartesian_prod but works with 0D, 1D and 2D tensors and returns 2D tensor.
    If tensor is 2D, its rows are considered as one-piece elements of product, i.e. concated to the rows of the result cartesian product.

    Example:
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        cartesian_prod(a, b) = torch.tensor([[1, 2, 5, 6], [1, 2, 7, 8], [3, 4, 5, 6], [3, 4, 7, 8]])

    Args:
        tensors: list of 0D, 1D or 2D tensors

    Returns:
        2D tensor with cartesian product of tensors
    """
    # check that all tensors are 0D, 1D or 2D
    assert all([t.dim() <= 2 for t in tensors])
    # reshape all tensors to 2D
    tensors = [t.view(t.numel(), 1) if t.dim() != 2 else t for t in tensors]
    # determine the shape of grids
    shape = [t.shape[0] for t in tensors]
    view_shape = [1] * len(shape)
    # create grids
    grids = []
    for i,t in enumerate(tensors):
        view_shape[i] = -1
        grid = t.view(*view_shape, t.shape[-1]).expand(*shape, t.shape[-1])
        view_shape[i] = 1
        grids.append(grid)
    prod = torch.cat(grids, dim=-1)
    prod = prod.view(-1, prod.shape[-1])
    return prod


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
