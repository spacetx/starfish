from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import xarray as xr

from ._base import FilterAlgorithmBase
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip
from starfish.core.util import click
from starfish.types import Axes


class CallBases(FilterAlgorithmBase):
    def __init__(self, intensity_threshold: float = 0, quality_threshold: float = 0,
        is_volume: bool = False, expand_dynamic_range: bool = False,
        clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:

        self.intensity_threshold = intensity_threshold
        self.quality_threshold = quality_threshold
        self.is_volume = is_volume
        self.expand_dynamic_range = expand_dynamic_range
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"intensity_threshold": 0, "quality_threshold": 0}

    def _vector_norm(self, x, dim, ord=None):
        return xr.apply_ufunc(np.linalg.norm, x,
                              input_core_dims=[[dim]],
                              kwargs={'ord': ord, 'axis': -1})

    def _call_bases(
        self, image: xr.DataArray, intensity_threshold: float,
        quality_threshold: float
    ) -> xr.DataArray:

        # Get the maximum value for each round/z
        max_chan = image.argmax(dim=Axes.CH.value)
        max_values = image.max(dim=Axes.CH.value)

        # Get the norms for each pixel
        norms = self._vector_norm(x=image, dim=Axes.CH.value)

        # Calculate the base qualities
        base_qualities = max_values / norms

        # Filter the base call qualities
        base_qualities_filtered = xr.where(base_qualities < quality_threshold,
            0, base_qualities)

        # Threshold the intensity values
        base_qualities_filtered = xr.where(max_values < intensity_threshold,
            0, base_qualities_filtered)

        # Put the base calls in place
        base_calls = xr.full_like(other=image, fill_value=0)
        base_calls[max_chan] = base_qualities_filtered

        return base_calls

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

        group_by = {Axes.ROUND, Axes.ZPLANE}
        unmix = partial(
            self._call_bases, intensity_threshold=self.intensity_threshold,
            quality_threshold=self.quality_threshold
        )
        result = stack.apply(
            unmix,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method,
        )
        return result

    @staticmethod
    @click.command("CallBases")
    @click.option(
        "--int-thresh", default=0, type=float, help="Intensity threshold for a base call")
    @click.option(
        "--qual-thresh", default=0, type=float, help="Quality threshold for a base call")
    @click.option(
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.option(
        "--expand-dynamic-range", is_flag=True,
        help="linearly scale data to fill [0, 1] after clipping."
    )
    @click.pass_context
    def _cli(ctx, p_min, p_max, is_volume, expand_dynamic_range):
        ctx.obj["component"]._cli_run(ctx, CallBases(int_thresh, qual_thresh, is_volume, expand_dynamic_range))
