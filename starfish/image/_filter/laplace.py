# Definition of the processing class
import argparse
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_laplace

from starfish.image._filter._base import FilterAlgorithmBase
from starfish.image._filter.util import (
    determine_axes_to_split_by,
    preserve_float_range,
    validate_and_broadcast_kernel_size,
)
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Number


class Laplace(FilterAlgorithmBase):
    """
    Multi-dimensional Laplace filter, using Gaussian second derivatives.
    This filter wraps scipy.ndimage.gaussian_laplace

    Parameters
    ----------
    sigma : Union[Number, Tuple[Number]]
        Standard deviation for Gaussian kernel.
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
        independently.

    """

    def __init__(
            self,
            sigma: Union[Number, Tuple[Number]], mode: str='reflect',
            cval: float=0.0, is_volume: bool=False, **kwargs
    ) -> None:
        """Multi-dimensional gaussian-laplacian filter used to enhance dots against background

        Parameters
        ----------
        sigma_gauss : Union[Number, Tuple[Number]]
            Standard deviation for Gaussian kernel to enhance dots.

        mode: The mode parameter determines how the input array is extended when
            the filter overlaps a border. By passing a sequence of modes with
            length equal to the number of dimensions of the input array,
            different modes can be specified along each axis. Default value
            is ‘reflect’.
            The valid values and their behavior is as follows:

            ‘reflect’ (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last pixel.

            ‘constant’ (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the same
            constant value, defined by the cval parameter.

            ‘nearest’ (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.

            ‘mirror’ (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last pixel.

            ‘wrap’ (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.

        cval : scalar, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

        is_volume: bool
            True is the image is a stack
        """

        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume=is_volume)
        self.mode = mode
        self.cval = cval
        self.is_volume = is_volume

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 0.5}

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float,
            help="Standard deviation of gaussian kernel for spot enhancement")
        group_parser.add_argument(
            "--mode", default="reflect",
            help="How the input array is extended when the filter overlaps a border")
        group_parser.add_argument(
            "--cval", default=0.0,
            help="Value to fill past edges of input if mode is ‘constant")
        group_parser.add_argument(
            "--is-volume", action="store_true",
            help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def _gaussian_laplace(image: np.ndarray, sigma: Union[Number, Tuple[Number]],
                          mode: str = 'reflect', cval: float = 0.0) -> np.ndarray:
        filtered = gaussian_laplace(
            image, sigma=sigma, mode=mode, cval=cval)

        filtered = -filtered  # the peaks are negative so invert the signal
        filtered = preserve_float_range(filtered)

        return filtered

    def run(
            self, stack: ImageStack, in_place: bool = True, verbose: bool = True,
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
        Returns
        -------
        ImageStack :
            if in-place is False, return the results of filter as a new stack
        """
        split_by = determine_axes_to_split_by(self.is_volume)
        apply_filtering: Callable = partial(self._gaussian_laplace, sigma=self.sigma)
        return stack.apply(
            apply_filtering,
            split_by=split_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
        )
