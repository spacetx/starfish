from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage.filters import uniform_filter

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number
from ._base import FilterAlgorithm
from .util import (
    determine_axes_to_group_by, validate_and_broadcast_kernel_size
)


class MeanHighPass(FilterAlgorithm):
    """
    The mean high pass filter reduces low spatial frequency features by subtracting a
    mean filtered image from the original image. The mean filter smooths an image by replacing
    each pixel's value with an average of the pixel values of the surrounding neighborhood.

    The mean filter is also known as a uniform or box filter. It can also be considered as a fast
    approximation to a GaussianHighPass filter.

    This is a pass through for :py:func:`scipy.ndimage.filters.uniform_filter`

    Parameters
    ----------
    size : Union[Number, Tuple[Number]]
        width of the kernel
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
        independently.
    level_method : :py:class:`~starfish.types.Levels`
        Controls the way that data are scaled to retain skimage dtype requirements that float data
        fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Levels.CLIP (default): data above 1 are set to 1.
        - Levels.SCALE_SATURATED_BY_IMAGE: when any data in the entire ImageStack is greater
          than 1, the entire ImageStack is scaled by the maximum value in the ImageStack.
        - Levels.SCALE_SATURATED_BY_CHUNK: when any data in any slice is greater than 1, each
          slice is scaled by the maximum value found in that slice.  The slice shapes are
          determined by the ``group_by`` parameters.
        - Levels.SCALE_BY_IMAGE: scale the entire ImageStack by the maximum value in the
          ImageStack.
        - Levels.SCALE_BY_CHUNK: scale each slice by the maximum value found in that slice.  The
          slice shapes are determined by the ``group_by`` parameters.
    """

    def __init__(
            self,
            size: Union[Number, Tuple[Number]],
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP
    ) -> None:

        self.size = validate_and_broadcast_kernel_size(size, is_volume)
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"size": 1}

    @staticmethod
    def _high_pass(
        image: xr.DataArray, size: Number, rescale: bool = False
    ) -> xr.DataArray:
        """
        Applies a mean high pass filter to an image

        Parameters
        ----------
        image : xr.DataArray
            2-d or 3-d image data
        size : Union[Number, Tuple[Number]]
            width of the kernel
        rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        xr.DataArray:
            Filtered image, same shape as input
        """

        blurred: np.ndarray = uniform_filter(image, size)

        filtered: xr.DataArray = image - blurred

        return filtered

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to applying the filter. If None, defaults to
            the result of os.cpu_count(). (default None)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        high_pass: Callable = partial(self._high_pass, size=self.size)
        result = stack.apply(
            high_pass,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            level_method=self.level_method
        )
        return result
