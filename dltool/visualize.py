import numpy as np
import torch
from collections.abc import Sequence


def scale(t: torch.Tensor, min_value: float = 0, max_value: float = 1,
          dim: int | Sequence[int] = None, q: float = None) -> torch.Tensor:
    """
    Scales the tensor values so that they lie between specified boundary values.

    Args:
        t: initial tensor
        min_value: desired minimum value of the resulting tensor
        max_value: desired maximum value of the resulting tensor
        dim: dimensions to reduce when calculating the minimum and maximum,
            i.e. slices along the other dimensions will be viewed independently;
            if None, the same transformation is applied to all tensor elements
        q: lower quantile, if given, the data between the q-th and (1-q)-th quantiles
            will be normalized. Values outside will be clipped.

    Returns:
        Scaled tensor
    """
    if q is not None:
        if dim is None:
            dim = tuple(range(t.dim()))
        if isinstance(dim, int):
            dim = (dim,)
        left_positions = tuple(range(len(dim)))
        t = t.movedim(dim, left_positions)
        t_shape = t.size()
        t = t.flatten(end_dim=len(dim)-1)
        lower_q, upper_q = torch.quantile(t, q=torch.tensor([q, 1-q], device=t.device), dim=0, keepdim=True)
        t = t.clamp(min=lower_q, max=upper_q)
        t = t.reshape(t_shape).movedim(left_positions, dim)
    t_min, t_max = t.amin(dim=dim, keepdim=True), t.amax(dim=dim, keepdim=True)
    t = (t - t_min) * (max_value - min_value) / (t_max - t_min + 1e-5) + min_value
    return t


@torch.no_grad()
def make_grid(
        t: torch.Tensor,
        ncols: int = 8,
        pad: Sequence | int = 1,
        pad_value: Sequence | float = 0,
        normalize: bool = False,
        quantile: float = None,
        channel_dim: int = 1,
        return_numpy: bool = True
) -> np.ndarray:
    """
    Makes grid from batch of images with default shape (n, channels, height, width).
    Indicate the channel dim if its order is different.

    Args:
        t: initial tensor with shape (n, channels, height, width)
        ncols: number of columns in the resulting grid
        pad: padding between images
        pad_value: value to use for padding
        normalize: if True, the tensor values will be scaled to [0,1] range independently
            for each channel
        quantile: lower quantile, if given and normalization is enabled, the data values
            will be clipped to fall in the interval between the q-th and (1-q)-th quantiles
        channel_dim: channel dimension index
        return_numpy: if True, the result will be returned as a numpy array,
            and channels will be the last dimension

    Returns:
        Grid tensor or array
    """
    if normalize is True:
        t = scale(t, dim=[i for i in range(t.dim()) if i != channel_dim], q=quantile)
    t = t.movedim(channel_dim, 1)
    # add padding
    if isinstance(pad, int):
        padding = [pad, 0, pad, 0]
    else:
        padding = [pad[0], 0, pad[1], 0]
    t = torch.nn.functional.pad(t, padding, 'constant', pad_value)
    # make grid
    nindex, channels, height, width = t.shape
    ncols = min(nindex, ncols)
    nrows = (nindex + ncols - 1) // ncols
    r = nrows * ncols - nindex  # remainder
    # want result.shape = (channels, nrows*height, ncols*width)
    arr = torch.cat([t] + [torch.full([1, channels, height, width], pad_value)] * r)
    result = (arr.reshape(nrows, ncols, channels, height, width)
              .permute(2, 0, 3, 1, 4)
              .reshape(channels, nrows * height, ncols * width))
    result = torch.nn.functional.pad(result, [0, padding[0], 0, padding[2]], 'constant', pad_value)
    if return_numpy:
        return result.permute(1, 2, 0).cpu().numpy()
    return result
