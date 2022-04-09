import cv2
import numpy as np
import numbers


def put_caption(img, text):
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


def img_grid(t_array, normalize=False, ncols=8, captions=None, pad=1):
    """
    Makes grid from batch of images with shape (n_batch, channels, height, width)
    """
    array = t_array.detach().permute(0, 2, 3, 1)
    if normalize is True:
        amin, amax = array.amin(dim=[0, 1, 2]), array.amax(dim=[0, 1, 2])
        array = (array - amin) / (amax - amin + 1e-5)
    # add padding
    padding = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
    array = np.pad(array.numpy(), padding, 'constant')
    # captions
    if captions is not None:
        for i in range(array.shape[0]):
            put_caption(array[i], captions[i])
    # make grid
    nindex, height, width, intensity = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex + ncols - 1) // ncols
    r = nrows * ncols - nindex  # remainder
    # want result.shape = (height*nrows, width*ncols, intensity)
    arr = np.concatenate([array] + [np.zeros([1, height, width, intensity])] * r)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return np.pad(result, [(pad, pad), (pad, pad), (0, 0)], 'constant')