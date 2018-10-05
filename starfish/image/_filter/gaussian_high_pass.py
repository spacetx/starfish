import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from starfish.image._filter.gaussian_low_pass import GaussianLowPass
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Number
from ._base import FilterAlgorithmBase
from .util import (
    determine_axes_to_split_by,
    preserve_float_range,
    validate_and_broadcast_kernel_size,
)


class GaussianHighPass(FilterAlgorithmBase):

    def __init__(
            self, sigma: Union[Number, Tuple[Number]], is_volume: bool=False, **kwargs
    ) -> None:
        """Gaussian high pass filter

        Parameters
        ----------
        sigma : Union[Number, Tuple[Number]]
            standard deviation of gaussian kernel
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
            independently.

        """
        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume)
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 3}

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float, help="standard deviation of gaussian kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def _high_pass(
            image: Union[xr.DataArray, np.ndarray],
            sigma: Union[Number, Tuple[Number]],
            rescale: bool=False
    ) -> Union[xr.DataArray, np.ndarray]:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        image : numpy.ndarray[np.float32]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of gaussian kernel
        rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        np.ndarray :
            filtered image of the same shape as the input image
        """

        blurred = GaussianLowPass._low_pass(image, sigma)
        filtered = image - blurred
        filtered = preserve_float_range(filtered, rescale)

        return filtered

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=True,
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
        high_pass: Callable = partial(self._high_pass, sigma=self.sigma)
        result = stack.apply(
            high_pass,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
