from functools import partial
from typing import Optional, Set

import numpy as np
import xarray as xr

from starfish.core.compat import match_histograms
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from starfish.core.util import enum
from ._base import FilterAlgorithm


class MatchHistograms(FilterAlgorithm):
    """
    Normalize data by matching distributions of each tile or volume to a reference volume

    Chunks sharing the same values for axes specified by group_by will be quantile
    normalized such that their intensity values are identically distributed. The reference
    distribution is calculated by sorting the intensities in each chunk and averaging across
    chunks.

    For example, if group_by={Axes.CH, Axes.ROUND} each (z, y, x) volume will be linearized,
    the intensities will be sorted, and averaged across {Axes.CH, Axes.ROUND}, normalizing
    the intensity distribution of each (round, channel) volume.

    Setting group_by={Axes.CH} would carry out the same approach, but the result would equalize
    distribution across channels only and would retain variability across rounds.

    Parameters
    ----------
    group_by : Set[Axes]
        The Axes to group by.
    """

    def __init__(
            self, group_by: Set[Axes]
    ) -> None:

        self.group_by = group_by

    _DEFAULT_TESTING_PARAMETERS = {"group_by": {Axes.CH, Axes.ROUND}}

    def _compute_reference_distribution(self, data: ImageStack) -> xr.DataArray:
        """compute the average reference distribution across the ImageStack"""
        chunk_key = enum.harmonize(data.shape.keys() - self.group_by)
        sort_key = enum.harmonize(self.group_by)

        # stack up the array
        stacked = data.xarray.stack(chunk_key=chunk_key)
        stacked = stacked.stack(sort_key=sort_key)

        sorted_stacked = stacked.groupby("sort_key").map(np.sort)
        reference = sorted_stacked.mean("sort_key")
        reference = reference.unstack("chunk_key")
        return reference

    @staticmethod
    def _match_histograms(
        image: xr.DataArray, reference: np.ndarray
    ) -> np.ndarray:
        """
        matches the intensity distribution of image to reference

        Parameters
        ----------
        image, reference : xr.DataArray
            3-d image data

        Returns
        -------
        np.ndarray :
            image, with intensities matched to reference
        """
        image_array = image.data
        reference = np.asarray(reference)
        return match_histograms(image_array, reference=reference)

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
        if verbose:
            print("Calculating reference distribution...")
        reference_image = self._compute_reference_distribution(stack)
        apply_function = partial(self._match_histograms, reference=reference_image)
        result = stack.apply(
            apply_function,
            group_by=self.group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
