import argparse
from functools import partial
from typing import Callable, Optional

import numpy as np

from starfish.image import ImageStack
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
from ._base import FilterAlgorithmBase


class GaussianHighPass(FilterAlgorithmBase):

    def __init__(self, sigma, **kwargs) -> None:
        """Gaussian high pass filter

        Parameters
        ----------
        sigma : int (default = 1)
            standard deviation of gaussian kernel

        """
        self.sigma = sigma

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", default=1, type=int, help="standard deviation of gaussian kernel")

    @staticmethod
    def gaussian_high_pass(img: np.ndarray, sigma) -> np.ndarray:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        img : numpy.ndarray
            Image to filter
        sigma : Union[float, int]
            Standard deviation of gaussian kernel

        Returns
        -------
        numpy.ndarray :
            Filtered image, same shape as input

        """
        blurred: np.ndarray = GaussianLowPass.low_pass(img, sigma)

        over_flow_ind: np.ndarray[bool] = img < blurred
        res: np.ndarray = img - blurred
        res[over_flow_ind] = 0

        return res

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
        high_pass: Callable = partial(self.gaussian_high_pass, sigma=self.sigma)
        result = stack.apply(high_pass, in_place=in_place)
        if not in_place:
            return result
        return None
