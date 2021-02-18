import warnings
from functools import partial
from typing import Optional

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels
from ._base import FilterAlgorithm
from .util import determine_axes_to_group_by


class Clip(FilterAlgorithm):
    """
    Image clipping filter that clips values below a minimum percentile and above a maximum
    percentile.

    By default, these min and max percentiles are set to 0 and 100 respectively, which will
    result in the filter doing nothing.

    This is a wrapper for :py:func:`numpy.clip` that can optionally linearly expand the dynamic
    range of the data to extend from [0, 1]

    Parameters
    ----------
    p_min : int
        values below this percentile are set to p_min (default 0)
    p_max : int
        values above this percentile are set to p_max (default 100)
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) tiles
    expand_dynamic_range : bool
        If True, linearly expand intensity values to fill [0, 1] after clipping.  This has been
        deprecated in favor of the level_method argument.  If ``expand_dynamic_range`` is True and
        ``level_method`` is provided, an error is raised.  If ``expand_dynamic_range`` is True and
        ``level_method`` is not provided, it is interpreted as
        ``level_method=Levels.SCALE_BY_CHUNK``.  If ``expand_dynamic_range`` is False or not
        provided and ``level_method`` is provided, ``level_method`` is used.  If neither is
        provided, it is interpreted as ``level_method=Levels.CLIP``. (default False)
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
            self, p_min: int = 0, p_max: int = 100, is_volume: bool = False,
            expand_dynamic_range: Optional[bool] = None,
            level_method: Optional[Levels] = None,
    ) -> None:

        self.p_min = p_min
        self.p_max = p_max
        self.is_volume = is_volume
        if expand_dynamic_range is not None:
            if level_method is not None:
                raise ValueError(
                    "Cannot provide both expand_dynamic_range and level_method."
                )
            warnings.warn(
                "Parameter `expand_dynamic_range` is deprecated.  Please use the level_method "
                "instead."
            )
            self.level_method = Levels.SCALE_BY_CHUNK
        else:
            if level_method is not None:
                self.level_method = level_method
            else:
                self.level_method = Levels.CLIP

    _DEFAULT_TESTING_PARAMETERS = {"p_min": 0, "p_max": 100}

    @staticmethod
    def _clip(
            image: xr.DataArray, p_min: int, p_max: int,
    ) -> xr.DataArray:
        """Clip values of image"""
        v_min, v_max = np.percentile(image, [p_min, p_max])

        image = image.clip(min=v_min, max=v_max)

        return image

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
        clip = partial(self._clip, p_min=self.p_min, p_max=self.p_max)
        result = stack.apply(
            clip,
            group_by=group_by,
            verbose=verbose,
            in_place=in_place,
            n_processes=n_processes,
            level_method=self.level_method,
        )
        return result
