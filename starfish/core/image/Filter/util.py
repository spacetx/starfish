from typing import Set, Tuple, Union

import numpy as np

from starfish.core.types import Axes, Number


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


def validate_and_broadcast_kernel_size(
        sigma: Union[Number, Tuple[Number, ...]],
        is_volume: bool
) -> Tuple[Number, ...]:
    """
    Check that the provided sigma is of the right dimensionality, and if necessary, broadcast it
    to full dimensionality.

    Parameters
    ----------
    sigma : Union[Number, Tuple[Number]]
    is_volume : bool

    Returns
    -------
    Tuple[Number] :
        2-d or 3-d kernel size.

    """
    if isinstance(sigma, tuple):
        message = ("if passing an anisotropic kernel, the dimensionality must match the data "
                   "shape ({shape}), not {passed_shape}")
        if is_volume and len(sigma) != 3:
            raise ValueError(message.format(shape=3, passed_shape=len(sigma)))
        if not is_volume and len(sigma) != 2:
            raise ValueError(message.format(shape=2, passed_shape=len(sigma)))
        valid_sigma = sigma
    else:
        if is_volume:
            valid_sigma = (sigma,) * 3
        else:
            valid_sigma = (sigma,) * 2

    return valid_sigma


def determine_axes_to_group_by(is_volume: bool) -> Set[Axes]:
    """map is_volume to axes to group by when applying a function over an ImageStack"""
    if is_volume:
        return {Axes.ROUND, Axes.CH}
    else:
        return {Axes.ROUND, Axes.CH, Axes.ZPLANE}
