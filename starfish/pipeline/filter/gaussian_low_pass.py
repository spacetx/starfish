from functools import partial
from typing import Optional

import numpy
from skimage.filters import gaussian

from starfish.image import ImageStack
from ._base import FilterAlgorithmBase


# TODO ambrosejcarr: need a better solution for 2d/3d support for image analysis algorithms

class GaussianLowPass(FilterAlgorithmBase):

    def __init__(self, sigma, is_volume: bool=False, verbose=False, **kwargs) -> None:
        """Multi-dimensional low-pass gaussian filter.

        Parameters
        ----------
        sigma : Union[float, Tuple[float]]
            Standard deviation for Gaussian kernel.
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles independently.
        verbose : bool
            If True, report on the percentage completed (default = False) during processing

        """
        self.sigma = sigma
        self.is_volume = is_volume
        self.verbose = verbose

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--sigma", default=1, type=int, help="standard deviation of gaussian kernel")

    @staticmethod
    def low_pass(image, sigma) -> numpy.ndarray:
        """Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray
            Image data
        sigma : Union[float, int, Tuple]
            Standard deviation of the Gaussian kernel that will be applied. If a float, an isotropic kernel will be
            assumed, otherwise the dimensions of the kernel give (z, x, y)

        Returns
        -------
        np.ndarray :
            Blurred data in same shape as input image.

        """
        blurred = gaussian(
            image, sigma=sigma, output=None, cval=0, multichannel=True, preserve_range=True, truncate=4.0)

        blurred = blurred.astype(numpy.uint16)

        return blurred

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
        low_pass = partial(self.low_pass, sigma=self.sigma)
        result = stack.apply(low_pass, is_volume=self.is_volume, verbose=self.verbose, in_place=in_place)
        if not in_place:
            return result
        return None
