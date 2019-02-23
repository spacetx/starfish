from functools import partial
from typing import Mapping, Optional, Union

import numpy as np
import xarray as xr

from starfish.compat import match_histograms
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class MatchHistograms(FilterAlgorithmBase):

    def __init__(
            self, reference_selector: Mapping[Axes, int],
    ) -> None:
        """Normalize data by matching distributions of each tile or volume to a reference volume

        Parameters
        ----------
        reference_selector : Mapping[Axes, int]
            A mapping that specifies the round and channel to match the intensity of each subsequent
            image to. For example, {Axes.CH: 0, Axes.ROUND: 0} would match each image to the first
            round and channel. This filter automatically detects whether the data is flat or
            volumetric.
        """
        self.reference_selector = reference_selector

    _DEFAULT_TESTING_PARAMETERS = {"reference_selector": {Axes.CH: 0, Axes.ROUND: 0}}

    @staticmethod
    def _match_histograms(
        image: Union[xr.DataArray, np.ndarray], reference: np.ndarray
    ) -> np.ndarray:
        """
        matches the intensity distribution of image to reference

        Parameters
        ----------
        image, reference : numpy.ndarray[np.float32]
            2-d or 3-d image data

        Returns
        -------
        np.ndarray :
            image, with intensities matched to reference
        """
        if isinstance(image, xr.DataArray):
            image = image.values
        return match_histograms(image, reference=reference)

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=True,
            n_processes: Optional[int]=None
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
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(is_volume=stack.shape[Axes.ZPLANE] > 1)
        reference_image: np.ndarray = stack.xarray.sel(self.reference_selector).values
        apply_function = partial(self._match_histograms, reference=reference_image)
        result = stack.apply(
            apply_function,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("MatchHistograms")
    @click.option(
        "--reference-selector", type=dict,
        help=("dict that specifies the round and channel to match the intensity of each"
              "subsequent image to, e.g. {Axes.CH: 0, Axes.ROUND: 0}")
    )
    @click.pass_context
    def _cli(ctx, reference_selector):
        ctx.obj["component"]._cli_run(ctx, MatchHistograms(reference_selector))
