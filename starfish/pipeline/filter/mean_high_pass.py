import argparse
from functools import partial
from numbers import Number
from typing import Callable, Union, Tuple, Optional

import numpy as np
from skimage import img_as_uint

from starfish.errors import DataFormatWarning
from starfish.image import ImageStack
from scipy.ndimage.filters import uniform_filter
from ._base import FilterAlgorithmBase


class MeanHighPass(FilterAlgorithmBase):

    def __init__(
            self, size: Number, is_volume: bool=False, verbose: bool=False, **kwargs
    ) -> None:
        """Mean high pass filter

        Parameters
        ----------
        size : Number
            width of the kernel
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles independently.
        verbose : bool
            if True, report on filtering progress (default = False)

        """

        self.size = size
        self.is_volume = is_volume
        self.verbose = verbose

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--size", type=float, help="width of the kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true", help="indicates that the image stack should be filtered in 3d")

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
            DataFormatWarning('mean filters currently only support uint16 images. Image data will be converted.')
            image = img_as_uint(image)

        blurred: np.ndarray = uniform_filter(image, size)

        over_flow_ind: np.ndarray[bool] = image < blurred
        filtered: np.ndarray = image - blurred
        filtered[over_flow_ind] = 0

        return filtered

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
            if in_place is False, return the results of filter as a new stack

        """
        high_pass: Callable = partial(self.high_pass, size=self.size)
        result = stack.apply(high_pass, is_volume=self.is_volume, verbose=self.verbose, in_place=in_place)
        if not in_place:
            return result
        return None
