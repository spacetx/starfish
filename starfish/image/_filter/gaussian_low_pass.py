import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from skimage import img_as_uint
from skimage.filters import gaussian

from starfish.errors import DataFormatWarning
from starfish.stack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase
from .util import validate_and_broadcast_kernel_size


class GaussianLowPass(FilterAlgorithmBase):

    def __init__(
            self, sigma: Union[Number, Tuple[Number]], is_volume: bool=False, **kwargs) -> None:
        """Multi-dimensional low-pass gaussian filter.

        Parameters
        ----------
        sigma : Union[Number, Tuple[Number]]
            Standard deviation for Gaussian kernel.
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
            independently.

        """
        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume)
        self.is_volume = is_volume

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float, help="standard deviation of gaussian kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def low_pass(image: np.ndarray, sigma: Union[Number, Tuple[Number]]) -> np.ndarray:
        """
        Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray[np.uint16]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of the Gaussian kernel that will be applied. If a float, an
            isotropic kernel will be assumed, otherwise the dimensions of the kernel give (z, y, x)

        Returns
        -------
        np.ndarray :
            Blurred data in same shape as input image, converted to np.uint16 dtype.

        """
        if image.dtype != np.uint16:
            DataFormatWarning(
                "gaussian filters only support uint16 images. Image data will be converted")
            image = img_as_uint(image)

        blurred = gaussian(
            image,
            sigma=sigma, output=None, cval=0, multichannel=False, preserve_range=True, truncate=4.0
        )
        blurred = blurred.clip(0).astype(np.uint16)

        return blurred

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
            if True, report on filtering progress (default = False)

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """
        low_pass: Callable = partial(self.low_pass, sigma=self.sigma)
        result = stack.apply(
            low_pass, is_volume=self.is_volume, verbose=verbose, in_place=in_place)
        if not in_place:
            return result
        return None
