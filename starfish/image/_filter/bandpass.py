from functools import partial
from typing import Optional

import numpy as np
from trackpy import bandpass

from starfish.stack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase


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
            # TODO dganguli: this is not documented by trackpy, can you help?
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) planes
        kwargs
        """
        self.lshort = lshort
        self.llong = llong
        self.threshold = threshold
        self.truncate = truncate
        self.is_volume = is_volume

    @classmethod
    def add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--lshort", type=float, help="filter signals below this frequency")
        group_parser.add_argument(
            "--llong", type=int, help="filter signals above this frequency")
        group_parser.add_argument(
            "--threshold", type=int, help="clip pixels below this intensity value")
        group_parser.add_argument("--truncate", default=4, type=int)  # TODO dganguli doc this too

    @staticmethod
    def bandpass(
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
             # TODO dganguli: this is not documented by trackpy, can you help?

        Returns
        -------
        np.ndarray :
            bandpassed image

        """
        bandpassed: np.ndarray = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed

    def run(
            self, stack: ImageStack, in_place: bool=True, verbose: bool=False
    ) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report the filtering progress across the tiles or volumes of the ImageStack

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """
        bandpass_ = partial(
            self.bandpass,
            lshort=self.lshort, llong=self.llong, threshold=self.threshold, truncate=self.truncate
        )
        result = stack.apply(
            bandpass_, verbose=verbose, in_place=in_place, is_volume=self.is_volume
        )
        if not in_place:
            return result
        return None
