import cv2
import numpy as np
import numbers
import torch
from typing import Union, Sequence


def scale(t: torch.Tensor, min_value: float = 0,
          max_value: float = 1, dim: Union[int, Sequence[int]] = None) -> torch.Tensor:
    """
    Scales the tensor values so that they lie between specified boundary values.

    Args:
        t: initial tensor
        min_value: desired minimum value of the resulting tensor
        max_value: desired maximum value of the resulting tensor
        dim: dimensions that will be reduced when calculating the minimum and maximum, i.e. tensor will be scaled
         independently along the other dimensions; if None, the same transformation is applied to all tensor elements

    Returns:
        Scaled tensor
    """
    if dim is None:
        dim = tuple(range(t.dim()))
    t_min, t_max = t.amin(dim=dim, keepdim=True), t.amax(dim=dim, keepdim=True)
    t = (t - t_min) * (max_value - min_value) / (t_max - t_min + 1e-5) + min_value
    return t


def put_caption(img: np.ndarray, text: str):
    """
    Places a small white colored caption in the upper left corner of the image array.

    Args:
        img: image array with shape (height,width,channels)
        text: caption text
    """
    if isinstance(text, numbers.Number):
        text = '{0:.2f}'.format(text)
    font_scale = 1.
    font = cv2.FONT_HERSHEY_PLAIN
    # set the rectangle background to black
    rectangle_bgr = (0., 0., 0.)
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = 0
    text_offset_y = 10
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 1, text_offset_y - text_height - 1))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(1., 1., 1.), thickness=1)


def make_grid(t: torch.Tensor, normalize: bool = False, ncols: int = 8, captions: Sequence[str] = None, pad: int = 1,
              channel_dim: int = 1) -> np.ndarray:
    """
    Makes grid from batch of images with default shape (n, channels, height, width).
    Indicate the channel dim if its order is different.

    Args:
        t: initial tensor
        normalize: whether to scale from 0 to 1
        ncols: number of columns in the resulting grid
        captions: tuple of captions for each image in the initial tensor
        pad: padding size, the images in the grid will be placed twice as far apart from each other
        channel_dim: channel dim location

    Returns:
        Image array
    """
    array = t.detach().movedim(channel_dim, -1)
    if normalize is True:
        array = scale(array, dim=tuple(range(array.dim() - 1)))
    # add padding
    padding = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
    array = np.pad(array.cpu().numpy(), padding, 'constant')
    # captions
    if captions is not None:
        for i in range(array.shape[0]):
            put_caption(array[i], captions[i])
    # make grid
    nindex, height, width, channels = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex + ncols - 1) // ncols
    r = nrows * ncols - nindex  # remainder
    # want result.shape = (height*nrows, width*ncols, channels)
    arr = np.concatenate([array] + [np.zeros([1, height, width, channels])] * r)
    result = (arr.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, channels))
    return np.pad(result, [(pad, pad), (pad, pad), (0, 0)], 'constant')
