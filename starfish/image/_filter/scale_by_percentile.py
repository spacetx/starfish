from functools import partial
from typing import Optional

import numpy as np

from starfish.stack import ImageStack
from ._base import FilterAlgorithmBase


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

    @classmethod
    def add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--p", default=100, type=int, help="scale images by this percentile")

    @staticmethod
    def scale(image: np.ndarray, p: int) -> np.ndarray:
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

        # asking for a float percentile clipping value from an integer image will
        # convert to float, so store the dtype so it can be restored
        dtype = image.dtype
        image = image / v
        return image.astype(dtype)

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
            If True, report on the percentage completed (default = False) during processing

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """
        clip = partial(self.scale, p=self.p)
        result = stack.apply(clip, is_volume=self.is_volume, verbose=verbose, in_place=in_place)
        if not in_place:
            return result
        return None
