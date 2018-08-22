from typing import Optional

import numpy as np
from skimage.morphology import ball, disk, white_tophat

from starfish.stack import ImageStack
from ._base import FilterAlgorithmBase


class WhiteTophat(FilterAlgorithmBase):
    """
    Performs "white top hat" filtering of an image to enhance spots. "White top hat filtering"
    finds spots that are both smaller and brighter than their surroundings.

    See Also
    --------
    https://en.wikipedia.org/wiki/Top-hat_transform
    """

    def __init__(self, masking_radius: int, is_volume: bool=False, **kwargs) -> None:
        """
        Instance of a white top hat morphological masking filter which masks objects larger
        than `masking_radius`

        Parameters
        ----------
        masking_radius : int
            radius of the morphological masking structure in pixels
        is_volume : int
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
            independently.

        """
        self.masking_radius = masking_radius
        self.is_volume = is_volume

    @classmethod
    def add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--masking-radius", default=15, type=int,
            help="diameter of morphological masking disk in pixels")

    def white_tophat(self, image: np.ndarray) -> np.ndarray:
        if self.is_volume:
            structuring_element = ball(self.masking_radius)
        else:
            structuring_element = disk(self.masking_radius)
        return white_tophat(image, selem=structuring_element)

    def run(
            self, stack: ImageStack, in_place: bool=True, verbose: bool=False) \
            -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            If True, report on the percentage completed (default = False) during processing

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """
        result = stack.apply(
            self.white_tophat, is_volume=self.is_volume, verbose=verbose, in_place=in_place)
        if not in_place:
            return result
        return None
