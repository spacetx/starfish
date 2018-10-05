from functools import partial
from typing import Optional

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_split_by, preserve_float_range


class ScaleByPercentile(FilterAlgorithmBase):

    def __init__(self, p: int=0, is_volume: bool=False, **kwargs) -> None:
        """Image scaling filter

        Parameters
        ----------
        p : int
            each image in the stack is scaled by this percentile. must be in [0, 100]
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) tiles

        kwargs
        """
        self.p = p
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"p": 0}

    @classmethod
    def _add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--p", default=100, type=int, help="scale images by this percentile")

    @staticmethod
    def _scale(image: np.ndarray, p: int) -> np.ndarray:
        """Clip values of img below and above percentiles p_min and p_max

        Parameters
        ----------
        image : np.ndarray
            image to be scaled

        p : int
            each image in the stack is scaled by this percentile. must be in [0, 100]

        Notes
        -----
        - Setting p to 100 scales the image by it's maximum value
        - No shifting or transformation to adjust dynamic range is done after scaling

        Returns
        -------
        np.ndarray :
          Numpy array of same shape as img

        """
        v = np.percentile(image, p)

        image = image / v
        image = preserve_float_range(image)

        return image

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
        clip = partial(self._scale, p=self.p)
        result = stack.apply(
            clip,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
