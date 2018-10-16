import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.ndimage.filters import uniform_filter

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase
from .util import (
    determine_axes_to_split_by, preserve_float_range, validate_and_broadcast_kernel_size
)


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

        self.size = validate_and_broadcast_kernel_size(size, is_volume)
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"size": 1}

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--size", type=float, help="width of the kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def _high_pass(image: np.ndarray, size: Number, rescale: bool=False) -> np.ndarray:
        """
        Applies a mean high pass filter to an image

        Parameters
        ----------
        image : numpy.ndarray[np.float32]
            2-d or 3-d image data
        size : Union[Number, Tuple[Number]]
            width of the kernel
        rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        np.ndarray [np.float32]:
            Filtered image, same shape as input
        """

        blurred: np.ndarray = uniform_filter(image, size)

        filtered: np.ndarray = image - blurred
        filtered = preserve_float_range(filtered, rescale)

        return filtered

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
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        split_by = determine_axes_to_split_by(self.is_volume)
        high_pass: Callable = partial(self._high_pass, size=self.size)
        result = stack.apply(
            high_pass,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
