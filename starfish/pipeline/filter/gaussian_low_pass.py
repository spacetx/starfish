from functools import partial
from typing import Union, Tuple

import numpy
from skimage.filters import gaussian

from ._base import FilterAlgorithmBase


class GaussianLowPass(FilterAlgorithmBase):

    def __init__(self, sigma: Union[Tuple[float], float], is_volume: bool=False, verbose=False, **kwargs) -> None:
        """Multi-dimensional low-pass gaussian filter.

        Parameters
        ----------
        sigma : float
            Standard deviation for Gaussian kernel.
        is_volume : bool
            If True, 3d (z, y, x) volumes will be passed to self.low_pass. Otherwise, 2d tiles will be passed
        verbose : bool
            If True, report on the percentage completed (default = False) during processing

        """
        self.sigma = sigma
        self.is_volume = is_volume
        self.verbose = verbose

    @classmethod
    def get_algorithm_name(cls):
        return "gaussian_low_pass"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--sigma", default=1, type=int, help="standard deviation of gaussian kernel")

    @staticmethod
    def low_pass(image, sigma: Union[float, Tuple[float]]) -> numpy.ndarray:
        """Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray
            Image data
        sigma : Union[float, Tuple[float]]
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

    def filter(self, stack) -> None:
        """Perform in-place filtering of an image stack and all contained aux images.

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered.

        """
        low_pass = partial(self.low_pass, sigma=self.sigma)
        stack.image.apply(low_pass, is_volume=self.is_volume, verbose=self.verbose)

        # apply to aux dict too:
        # TODO ambrosejcarr: tonytung's changes to the aux_dict -> imagestack will be super helpful here.
        # here, we take only the last two values of sigma, if passed, because the aux_dict is a 2d image, not a stack
        # this will change with tony's PR. [hack alert]
        for k, val in stack.aux_dict.items():
            stack.aux_dict[k] = self.low_pass(val, self.sigma[-2:])
