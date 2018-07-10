import argparse
from functools import partial
from typing import Callable, Optional

import numpy as np
from skimage import restoration

from starfish.image import ImageStack
from ._base import FilterAlgorithmBase
from .util import gaussian_kernel


class DeconvolvePSF(FilterAlgorithmBase):

    def __init__(
            self, num_iter: int=15, sigma: float=2, clip: bool=False, verbose=False, **kwargs
    ) -> None:
        """Deconvolve a point spread function

        Note that the default parameters are highly optimized for the MERFISH use case and that num_iter is a
        very important parameter that requires careful optimization.

        Parameters
        ----------
        num_iter : int
            number of iterations to run
        sigma : float
            standard deviation of the gaussian kernel used to construct the point spread function
        clip : bool (default = False)
            if True, pixel values below -1 and above 1 are clipped for skimage pipeline compatibility
        verbose : bool
            if True, report on the percentage completed during processing (default = False)

        """
        self.num_iter = num_iter
        self.sigma = sigma
        self.clip = clip
        self.kernel_size: int = int(2 * np.ceil(2 * sigma) + 1)
        self.psf: np.ndarray = gaussian_kernel(shape=(self.kernel_size, self.kernel_size), sigma=sigma)
        self.verbose = verbose

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument('--num-iter', default=15, type=int, help='number of iterations to run')
        group_parser.add_argument('--sigma', default=2, type=float, help='standard deviation of gaussian kernel')
        group_parser.add_argument(
            '--clip', action='store_true', help='(default False) if True, clip values below -1 and above 1')

    @staticmethod
    def richardson_lucy_deconv(img: np.ndarray, num_iter: int, psf: np.ndarray, clip: bool=False) -> np.ndarray:
        """
        Deconvolves input image with a specified point spread function. This simply calls
        skimage.restoration.richardson_lucy

        Parameters
        ----------
        img : np.ndarray
            Image to filter.
        num_iter : int
            Number of iterations to run algorithm
        psf :
            Point spread function
        clip : bool (default = False)
            If true, pixel value of the result above 1 or under -1 are thresholded for skimage pipeline compatibility.

        Returns
        -------
        np.ndarray :
            Deconvolved image, same shape as input

        """

        # TODO ambrosejcarr: the restoration function is producing the following warning:
        # /usr/local/lib/python3.6/site-packages/skimage/restoration/deconvolution.py:389: RuntimeWarning: invalid value
        # encountered in true_divide:
        # relative_blur = image / convolve_method(im_deconv, psf, 'same')
        img_deconv: np.ndarray = restoration.richardson_lucy(img, psf, iterations=num_iter, clip=clip)

        # here be dragons. img_deconv is a float. this should not work, but the result looks nice
        # modulo boundary values? wtf indeed.
        img_deconv = img_deconv.astype(np.uint16)
        return img_deconv

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
        func: Callable = partial(self.richardson_lucy_deconv, num_iter=self.num_iter, psf=self.psf, clip=self.clip)
        result = stack.apply(func, in_place=in_place, verbose=self.verbose)
        if not in_place:
            return result
        return None
