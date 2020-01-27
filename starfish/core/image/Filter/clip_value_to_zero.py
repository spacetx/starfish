from functools import partial
from typing import Optional

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number
from ._base import FilterAlgorithm
from .util import determine_axes_to_group_by


class ClipValueToZero(FilterAlgorithm):
    """
    Image clipping filter that clips values below a minimum value and above a
    maximum value. The filter then subtracts the minimum value from the
    clipped image.

    By default, the min and max values are set to 0.0 and None respectively,
    which will result in the filter doing nothing.

    This is a wrapper for :py:func:`numpy.clip`.

    Parameters
    ----------
    v_min : float
        Values below v_min are set to v_min. v_min is then subtracted from the
        entire image (default 0)
    v_max : Optional[Number]
        Values above v_max are set to v_max (default None)
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered. By default, filter 2D
        (y, x) tiles
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
            v_min: float = 0.0,
            v_max: Optional[Number] = None,
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP,
    ) -> None:
        self.v_min = v_min
        self.v_max = v_max
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"v_min": 0.0, "v_max": None}

    @staticmethod
    def _clip_value_to_zero(
            image: xr.DataArray,
            v_min: float,
            v_max: Optional[Number]) -> xr.DataArray:
        return image.clip(min=v_min, max=v_max) - np.float32(v_min)

    def run(self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args) -> Optional[ImageStack]:
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
            Number of parallel processes to devote to applying the filter.
            If None, defaults to the result of os.cpu_count(). (default None)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.
            Otherwise return the original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        clip_value_to_zero = partial(
            self._clip_value_to_zero,
            v_min=self.v_min, v_max=self.v_max,
        )
        result = stack.apply(
            clip_value_to_zero,
            group_by=group_by,
            verbose=verbose,
            in_place=in_place,
            n_processes=n_processes,
            level_method=self.level_method,
        )
        return result
