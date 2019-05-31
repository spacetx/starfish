from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Number
from starfish.core.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class ClipValueToZero(FilterAlgorithmBase):
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
    """

    def __init__(self,
                 v_min: float = 0.0,
                 v_max: Optional[Number] = None,
                 is_volume: bool = False) -> None:
        self.v_min = v_min
        self.v_max = v_max
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"v_min": 0.0, "v_max": None}

    @staticmethod
    def _clip_value_to_zero(image: Union[xr.DataArray, np.ndarray],
                            v_min: float,
                            v_max: Optional[Number]) -> np.ndarray:
        return image.clip(min=v_min, max=v_max) - np.float32(v_min)

    def run(self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args) -> ImageStack:
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
            group_by=group_by, verbose=verbose,
            in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("ClipValueToZero")
    @click.option(
        "--v-min", default=0.0, type=float,
        help=("clip intensities below this value and subtract this value "
              "from the image"))
    @click.option(
        "--v-max", default=None, type=float,
        help="clip intensities above this value")
    @click.option(
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.pass_context
    def _cli(ctx, v_min, v_max, is_volume):
        ctx.obj["component"]._cli_run(ctx,
                                      ClipValueToZero(v_min, v_max, is_volume))
