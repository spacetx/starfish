from functools import partial
from typing import Optional

import xarray as xr
from trackpy import bandpass

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number
from ._base import FilterAlgorithm
from .util import determine_axes_to_group_by


class Bandpass(FilterAlgorithm):
    """
    Convolve with a Gaussian to remove short-wavelength noise and subtract out long-wavelength
    variations, retaining features of intermediate scale. This implementation relies on
    scipy.ndimage.filters.gaussian_filter.

    This method is a thin wrapper around :doc:`trackpy:generated/trackpy.preprocessing.bandpass`.

    Parameters
    ----------
    lshort : float
        filter frequencies below this value
    llong : int
        filter frequencies above this odd integer value
    threshold : float
        zero any spots below this intensity value after background subtraction (default 0)
    truncate : float
        truncate the gaussian kernel, used by the gaussian filter, at this many standard
        deviations (default 4)
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) planes
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
            lshort: Number,
            llong: int,
            threshold: Number = 0,
            truncate: Number = 4,
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP
    ) -> None:
        self.lshort = lshort
        self.llong = llong

        if threshold is None:
            raise ValueError("Threshold cannot be None, please pass a float or integer")

        self.threshold = threshold
        self.truncate = truncate
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"lshort": 1, "llong": 3, "threshold": 0.01}

    @staticmethod
    def _bandpass(
            image: xr.DataArray,
            lshort: Number, llong: int, threshold: Number, truncate: Number
    ) -> xr.DataArray:
        """Apply a bandpass filter to remove noise and background variation

        Parameters
        ----------
        image : xr.DataArray
        lshort : float
            filter frequencies below this value
        llong : int
            filter frequencies above this odd integer value
        threshold : float
            zero any spots below this intensity value after background subtraction (default 0)
        truncate : float
            truncate the gaussian kernel, used by the gaussian filter, at this many standard
            deviations

        Returns
        -------
        xr.DataArray :
            bandpassed image

        """
        bandpassed = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed

    def run(
            self,
            stack: ImageStack,
            in_place: bool=False,
            verbose: bool=False,
            n_processes: Optional[int]=None,
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
        bandpass_ = partial(
            self._bandpass,
            lshort=self.lshort, llong=self.llong, threshold=self.threshold, truncate=self.truncate
        )

        group_by = determine_axes_to_group_by(self.is_volume)

        result = stack.apply(
            bandpass_,
            group_by=group_by,
            in_place=in_place,
            n_processes=n_processes,
            level_method=self.level_method,
            verbose=verbose,
        )
        return result
