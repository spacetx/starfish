from functools import partial
from typing import Optional

from trackpy import bandpass
import numpy as np

from starfish.image import ImageStack
from ._base import FilterAlgorithmBase


class Bandpass(FilterAlgorithmBase):

    def __init__(self, lshort, llong, threshold, truncate, verbose=False, **kwargs):
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
            # todo document
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        kwargs
        """
        self.lshort = lshort
        self.llong = llong
        self.threshold = threshold
        self.truncate = truncate
        self.verbose = verbose

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--lshort", default=0.5, type=float, help="filter signals below this frequency")
        group_parser.add_argument("--llong", default=7, type=int, help="filter signals above this frequency")
        group_parser.add_argument("--threshold", default=1, type=int, help="clip pixels below this intensity value")
        group_parser.add_argument("--truncate", default=4, type=int)

    @staticmethod
    def bandpass(image, lshort, llong, threshold, truncate) -> np.ndarray:
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
            # todo document

        Returns
        -------
        np.ndarray :
            bandpassed image

        """
        bandpassed = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed

    def filter(self, stack: ImageStack, in_place: bool=True) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """
        bandpass_ = partial(
            self.bandpass, lshort=self.lshort, llong=self.llong, threshold=self.threshold, truncate=self.truncate
        )
        result = stack.apply(bandpass_, verbose=self.verbose, in_place=in_place)
        if not in_place:
            return result
        return None
