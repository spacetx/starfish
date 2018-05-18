from functools import partial

import numpy
from skimage.filters import gaussian

from starfish.munge import swap
from ._base import FilterAlgorithmBase


class GaussianLowPass(FilterAlgorithmBase):

    def __init__(self, sigma, **kwargs):
        """Multi-dimensional low-pass gaussian filter.

        Parameters
        ----------
        sigma : float
            Standard deviation for Gaussian kernel.

        """
        self.sigma = sigma

    @classmethod
    def get_algorithm_name(cls):
        return "gaussian_low_pass"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--sigma", default=1, type=int, help="size of morphological masking disk")

    @staticmethod
    def low_pass(image, sigma):
        """Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray
            Image data
        sigma : float
            Standard deviation of the Gaussian kernel that will be applied.

        Returns
        -------
        np.ndarray :
            Blurred data in same shape as input image.

        """
        # TODO: ambrosejcarr what is the assumed axis order, and can we do away with swap?
        image_swap = swap(image)

        blurred = gaussian(
            image_swap, sigma=sigma, output=None, cval=0, multichannel=True, preserve_range=True, truncate=4.0)

        blurred = blurred.astype(numpy.uint16)

        return swap(blurred)

    def filter(self, stack) -> None:
        """Perform in-place filtering of an image stack and all contained aux images.

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered.

        """
        low_pass = partial(self.low_pass, sigma=self.sigma)
        stack.image.apply(low_pass)

        # apply to aux dict too:
        for k, val in stack.aux_dict.items():
            stack.aux_dict[k] = self.low_pass(val, self.sigma)
