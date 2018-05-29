from typing import Tuple

import numpy as np
from skimage.morphology import disk, binary_opening


def bin_thresh(img: np.ndarray, thresh: int) -> np.ndarray:
    """
    Performs binary thresholding of an image

    Parameters
    ----------
    img : np.ndarray
        Image to filter.
    thresh : int
        Pixel values >= thresh are set to 1, else 0.

    Returns
    -------
    np.ndarray :
        Binarized image, same shape as input

    """
    res = img >= thresh
    return res


def bin_open(img: np.ndarray, disk_size: int) -> np.ndarray:
    """
    Performs binary opening of an image

    img : np.ndarray
        Image to filter.
    disk_size : int
        Radius of the disk-shaped structuring element.

    Returns
    -------
    np.ndarray :
        Filtered image, same shape as input

    """
    selem = disk(disk_size)
    res = binary_opening(img, selem)
    return res


def gaussian_kernel(shape: Tuple[int, int]=(3, 3), sigma: float=0.5):
    """
    Returns a gaussian kernel of specified shape and standard deviation.
    This is a simple python implementation of Matlab's fspecial('gaussian',[shape],[sigma])

    Parameters
    ----------
    shape : Tuple[int] (default = (3, 3))
        Kernel shape.
    sigma : float (default = 0.5)
        Standard deviation of gaussian kernel.

    Parameters
    ----------
    np.ndarray :
        Gaussian kernel.
    """
    m, n = [int((ss - 1.) / 2.) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel
