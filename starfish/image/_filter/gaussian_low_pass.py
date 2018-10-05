import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from skimage.filters import gaussian

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase
from .util import (
    determine_axes_to_split_by,
    preserve_float_range,
    validate_and_broadcast_kernel_size,
)


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

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 1}

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float, help="standard deviation of gaussian kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def _low_pass(
            image: np.ndarray,
            sigma: Union[Number, Tuple[Number]],
            rescale: bool=False
    ) -> np.ndarray:
        """
        Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray[np.float32]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of the Gaussian kernel that will be applied. If a float, an
            isotropic kernel will be assumed, otherwise the dimensions of the kernel give (z, y, x)
         rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        np.ndarray :
            Blurred data in same shape as input image, converted to np.float32 dtype.

        """

        filtered = gaussian(
            image,
            sigma=sigma, output=None, cval=0, multichannel=False, preserve_range=True, truncate=4.0
        )

        filtered = preserve_float_range(filtered, rescale)

        return filtered

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=False,
            n_processes: Optional[int]=None,
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
        low_pass: Callable = partial(self._low_pass, sigma=self.sigma)
        result = stack.apply(
            low_pass,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
