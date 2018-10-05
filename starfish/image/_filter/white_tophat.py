from typing import Optional

import numpy as np
from skimage.morphology import ball, disk, white_tophat

from starfish.imagestack.imagestack import ImageStack
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_split_by


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

    _DEFAULT_TESTING_PARAMETERS = {"masking_radius": 3}

    @classmethod
    def _add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--masking-radius", default=15, type=int,
            help="diameter of morphological masking disk in pixels")

    def _white_tophat(self, image: np.ndarray) -> np.ndarray:
        if self.is_volume:
            structuring_element = ball(self.masking_radius)
        else:
            structuring_element = disk(self.masking_radius)
        return white_tophat(image, selem=structuring_element)

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
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        split_by = determine_axes_to_split_by(self.is_volume)
        result = stack.apply(
            self._white_tophat,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
