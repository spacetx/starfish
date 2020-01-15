from functools import partial
from typing import Callable, Optional, Tuple, Union

import xarray as xr

from starfish.core.image.Filter.gaussian_low_pass import GaussianLowPass
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number
from starfish.core.util.levels import levels
from ._base import FilterAlgorithm
from .util import (
    determine_axes_to_group_by,
    validate_and_broadcast_kernel_size,
)


class GaussianHighPass(FilterAlgorithm):
    """
    Applies a Gaussian high pass filter to the ImageStack. This is useful to remove cellular
    autofluorescence, which is typically low frequency.

    This filter works by subtracting a Gaussian low pass filter filtered version of the input image
    from the input image itself. The Gaussian low pass filter is defined in :py:func:`scipy.ndimage`
    and used by :py:func:`skimage.filters.gaussian`.

    Parameters
    ----------
    sigma : Union[Number, Tuple[Number]]
        standard deviation of gaussian kernel
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
            sigma: Union[Number, Tuple[Number]],
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP
    ) -> None:

        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume)
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 3}

    @staticmethod
    def _high_pass(
            image: xr.DataArray,
            sigma: Union[Number, Tuple[Number]],
    ) -> xr.DataArray:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        image : Union[xr.DataArray, np.ndarray]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of gaussian kernel
        rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        np.ndarray :
            filtered image of the same type and shape as the input image
        """

        blurred = GaussianLowPass._low_pass(image, sigma)
        blurred = levels(blurred)  # clip negative values to 0.
        filtered = image - blurred

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
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        high_pass: Callable = partial(self._high_pass, sigma=self.sigma)
        result = stack.apply(
            high_pass,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            level_method=self.level_method
        )
        return result
