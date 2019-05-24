from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class ClipPercentileToZero(FilterAlgorithmBase):
    """
    Image clipping filter that clips values below a minimum percentile and
    above a maximum percentile, and follows up by subtracting the minimum
    percentile value from the image.

    By default, these min and max percentiles are set to 0 and 100
    respectively, which will result in the filter doing nothing.

    This is a wrapper for :py:func:`numpy.clip`.

    Parameters
    ----------
    p_min : int
        Values below this percentile are set to p_min, and the p_min value
        is subtracted from the image (default 0)
    p_max : int
        Values above this percentile are set to p_max (default 100)
    min_coeff : float
        Apply a coefficient to the minimum percentile value. (default 1.0)
    max_coeff : float
        Apply a coefficient to the maximum percentile value. (default 1.0)
    is_volume : bool
        If True, 3D (z, y, x) volumes will be filtered. By default, filter 2D
        (y, x) tiles
    """

    def __init__(
        self, p_min: int = 0, p_max: int = 100,
        min_coeff: float = 1.0, max_coeff: float = 1.0,
        is_volume: bool = False,
    ) -> None:

        self.p_min = p_min
        self.p_max = p_max
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"p_min": 0, "p_max": 100,
                                   "min_coeff": 1.0, "max_coeff": 1.0}

    @staticmethod
    def _clip_percentile_to_zero(image: Union[xr.DataArray, np.ndarray],
                                 p_min: int,
                                 p_max: int,
                                 min_coeff: float,
                                 max_coeff: float) -> np.ndarray:
        v_min, v_max = np.percentile(image, [p_min, p_max])
        v_min = min_coeff * v_min
        v_max = max_coeff * v_max
        return image.clip(min=v_min, max=v_max) - np.float32(v_min)

    def run(self, stack: ImageStack, in_place: bool = False,
            verbose: bool = False, n_processes: Optional[int] = None,
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
        clip_percentile_to_zero = partial(
            self._clip_percentile_to_zero,
            p_min=self.p_min, p_max=self.p_max,
            min_coeff=self.min_coeff, max_coeff=self.max_coeff
        )
        result = stack.apply(
            clip_percentile_to_zero,
            group_by=group_by, verbose=verbose,
            in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("ClipPercentileToZero")
    @click.option(
        "--p-min", default=0, type=int,
        help=("clip intensities below this percentile and subtract the "
              "percentile value from the image"))
    @click.option(
        "--p-max", default=100, type=int,
        help="clip intensities above this percentile")
    @click.option(
        "--min-coeff", default=1.0, type=float,
        help="apply coefficient to minimum percentile value")
    @click.option(
        "--max-coeff", default=1.0, type=float,
        help="apply coefficient to maximum percentile value")
    @click.option(
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.pass_context
    def _cli(ctx, p_min, p_max, min_coeff, max_coeff, is_volume):
        ctx.obj["component"]._cli_run(ctx,
                                      ClipPercentileToZero(p_min, p_max,
                                                           min_coeff,
                                                           max_coeff,
                                                           is_volume))
