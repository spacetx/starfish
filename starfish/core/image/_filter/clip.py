from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class Clip(FilterAlgorithmBase):
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
        If True, linearly expand intensity values to fill [0, 1] after clipping. (default False)
    """

    def __init__(
        self, p_min: int = 0, p_max: int = 100, is_volume: bool = False,
        expand_dynamic_range: bool = False
    ) -> None:

        self.p_min = p_min
        self.p_max = p_max
        self.is_volume = is_volume
        self.expand_dynamic_range = expand_dynamic_range

    _DEFAULT_TESTING_PARAMETERS = {"p_min": 0, "p_max": 100}

    @staticmethod
    def _clip(
        image: Union[xr.DataArray, np.ndarray], p_min: int, p_max: int,
        expand_dynamic_range: bool
    ) -> np.ndarray:
        """Clip values of image"""
        v_min, v_max = np.percentile(image, [p_min, p_max])

        image = image.clip(min=v_min, max=v_max)
        if expand_dynamic_range:
            image /= np.max(image)

        return image

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> ImageStack:
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
        clip = partial(
            self._clip,
            p_min=self.p_min, p_max=self.p_max, expand_dynamic_range=self.expand_dynamic_range
        )
        result = stack.apply(
            clip,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("Clip")
    @click.option(
        "--p-min", default=0, type=int, help="clip intensities below this percentile")
    @click.option(
        "--p-max", default=100, type=int, help="clip intensities above this percentile")
    @click.option(
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.option(
        "--expand-dynamic-range", is_flag=True,
        help="linearly scale data to fill [0, 1] after clipping."
    )
    @click.pass_context
    def _cli(ctx, p_min, p_max, is_volume, expand_dynamic_range):
        ctx.obj["component"]._cli_run(ctx, Clip(p_min, p_max, is_volume, expand_dynamic_range))
