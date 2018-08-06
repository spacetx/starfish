from typing import Optional

import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from skimage.morphology import disk, ball

from starfish.image import ImageStack
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

    # TODO dganguli: any reason we're not using the white_tophat method?
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.\
    # morphology.white_tophat.html
    def white_tophat(self, image: np.ndarray) -> np.ndarray:
        """
        run a white top hat morphological masking filter which detects peaks that are smaller
        and brighter than their surroundings

        Parameters
        ----------
        image: np.ndarray
            2 or 3-d image

        Returns
        -------
        np.ndarray:
            filtered image

        """

        if image.dtype.kind != "u":
            raise TypeError("images should be stored in an unsigned integer array")

        if self.is_volume:
            structuring_element = ball(self.masking_radius)
        else:
            structuring_element = disk(self.masking_radius)

        min_filtered = minimum_filter(image, footprint=structuring_element)
        max_filtered = maximum_filter(min_filtered, footprint=structuring_element)
        filtered_image = image - np.minimum(image, max_filtered)
        return filtered_image

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
