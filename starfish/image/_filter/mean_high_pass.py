import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.ndimage.filters import uniform_filter
from skimage import img_as_uint

from starfish.errors import DataFormatWarning
from starfish.stack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase


class MeanHighPass(FilterAlgorithmBase):

    def __init__(
            self, size: Union[Number, Tuple[Number]], is_volume: bool=False, **kwargs) -> None:
        """Mean high pass filter.

        The mean high pass filter reduces low spatial frequency features by subtracting a
        mean filtered image from the original image. The mean filter smooths an image by replacing
        each pixel's value with an average of the pixel values of the surrounding neighborhood.

        The mean filter is also known as a uniform or box filter.

        This is a pass through for the scipy.ndimage.filters.uniform_filter:
        https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.ndimage.uniform_filter.html

        Parameters
        ----------
        size : Union[Number, Tuple[Number]]
            width of the kernel
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
            independently.

        """

        if isinstance(size, tuple):
            message = ("if passing an anisotropic kernel, the dimensionality must match the data "
                       "shape ({shape}), not {passed_shape}")
            if is_volume and len(size) != 3:
                raise ValueError(message.format(shape=3, passed_shape=len(size)))
            if not is_volume and len(size) != 2:
                raise ValueError(message.format(shape=2, passed_shape=len(size)))

        self.size = size
        self.is_volume = is_volume

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--size", type=float, help="width of the kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def high_pass(image: np.ndarray, size: Number) -> np.ndarray:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        image : numpy.ndarray[np.uint16]
            2-d or 3-d image data
        size : Number
            width of the kernel

        Returns
        -------
        np.ndarray :
            Filtered image, same shape as input

        """
        if image.dtype != np.uint16:
            DataFormatWarning(
                "Mean filters currently only support uint16 images. Image data will be converted.")
            image = img_as_uint(image)

        blurred: np.ndarray = uniform_filter(image, size)

        over_flow_ind: np.ndarray[bool] = image < blurred
        filtered: np.ndarray = image - blurred
        filtered[over_flow_ind] = 0

        return filtered

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
            if in_place is False, return the results of filter as a new stack

        """
        high_pass: Callable = partial(self.high_pass, size=self.size)
        result = stack.apply(
            high_pass, is_volume=self.is_volume, verbose=verbose, in_place=in_place)
        if not in_place:
            return result
        return None
