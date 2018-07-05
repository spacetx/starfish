from typing import Optional

import numpy as np

from starfish.image import ImageStack
from ._base import FilterAlgorithmBase


class WhiteTophat(FilterAlgorithmBase):
    """
    Performs "white top hat" filtering of an image to enhance spots. "White top hat filtering" finds spots that are both
    smaller and brighter than their surroundings.

    See Also
    --------
    https://en.wikipedia.org/wiki/Top-hat_transform
    """

    def __init__(self, disk_size, **kwargs):
        """Instance of a white top hat morphological masking filter which masks objects larger than `disk_size`

        Parameters
        ----------
        disk_size : int
            diameter of the morphological masking disk in pixels

        """
        self.disk_size = disk_size

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--disk-size", default=15, type=int, help="diameter of morphological masking disk in pixels")

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
        from scipy.ndimage.filters import maximum_filter, minimum_filter
        from skimage.morphology import disk

        def white_tophat(image):
            if image.dtype.kind != "u":
                raise TypeError("images should be stored in an unsigned integer array")
            structuring_element = disk(self.disk_size)
            min_filtered = minimum_filter(image, footprint=structuring_element)
            max_filtered = maximum_filter(min_filtered, footprint=structuring_element)
            filtered_image = image - np.minimum(image, max_filtered)
            return filtered_image

        result = stack.apply(white_tophat)
        if not in_place:
            return result
        return None
