from functools import partial
from typing import Optional

import numpy as np
from trackpy import bandpass

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_split_by


class Bandpass(FilterAlgorithmBase):

    def __init__(
            self, lshort: Number, llong: int, threshold: Number, truncate: Number=4,
            is_volume: bool=False, **kwargs) -> None:
        """

        Parameters
        ----------
        lshort : float
            filter frequencies below this value
        llong : int
            filter frequencies above this odd integer value
        threshold : float
            zero any pixels below this intensity value
        truncate : float
            truncate the gaussian kernel, used by the gaussian filter, at this many standard
            deviations
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) planes
        kwargs
        """
        self.lshort = lshort
        self.llong = llong
        self.threshold = threshold
        self.truncate = truncate
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"lshort": 1, "llong": 3, "threshold": 0.01}

    @classmethod
    def _add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--lshort", type=float, help="filter signals below this frequency")
        group_parser.add_argument(
            "--llong", type=int, help="filter signals above this frequency")
        group_parser.add_argument(
            "--threshold", type=int, help="clip pixels below this intensity value")
        group_parser.add_argument(
            "--truncate", default=4, type=float,
            help="truncate the filter at this many standard deviations")

    @staticmethod
    def _bandpass(
            image: np.ndarray, lshort: Number, llong: int, threshold: Number, truncate: Number
    ) -> np.ndarray:
        """Apply a bandpass filter to remove noise and background variation

        Parameters
        ----------
        image : np.ndarray
        lshort : float
            filter frequencies below this value
        llong : int
            filter frequencies above this odd integer value
        threshold : float
            zero any pixels below this intensity value
        truncate : float
            truncate the gaussian kernel, used by the gaussian filter, at this many standard
            deviations

        Returns
        -------
        np.ndarray :
            bandpassed image

        """
        bandpassed: np.ndarray = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed.astype(np.float32)

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=False,
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
            if True, report the filtering progress across the tiles or volumes of the ImageStack
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

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

        split_by = determine_axes_to_split_by(self.is_volume)

        result = stack.apply(
            bandpass_,
            split_by=split_by,
            in_place=in_place,
            n_processes=n_processes,
        )
        return result
