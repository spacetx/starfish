import argparse
from functools import partial
from typing import Optional

import numpy as np
from scipy.signal import convolve, fftconvolve

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Indices, Number
from ._base import FilterAlgorithmBase
from .util import gaussian_kernel, preserve_float_range


class DeconvolvePSF(FilterAlgorithmBase):

    def __init__(
            self, num_iter: int, sigma: Number, clip: bool=True, **kwargs) -> None:
        """Deconvolve a point spread function

        Parameters
        ----------
        num_iter : int
            number of iterations to run. Note that this is a very important parameter that requires
            careful optimization
        sigma : Number
            standard deviation of the gaussian kernel used to construct the point spread function
        clip : bool (default = False)
            if True, pixel values below -1 and above 1 are clipped for skimage pipeline
            compatibility

        """
        self.num_iter = num_iter
        self.sigma = sigma
        self.clip = clip
        self.kernel_size: int = int(2 * np.ceil(2 * sigma) + 1)
        self.psf: np.ndarray = gaussian_kernel(
            shape=(self.kernel_size, self.kernel_size),
            sigma=sigma
        )

    _DEFAULT_TESTING_PARAMETERS = {"num_iter": 1, "sigma": 1}

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            '--num-iter', type=int, help='number of iterations to run')
        group_parser.add_argument(
            '--sigma', type=float, help='standard deviation of gaussian kernel')
        group_parser.add_argument(
            '--no-clip', action='store_false',
            help='(default True) if True, clip values below 0 and above 1')

    # Here be dragons. This algorithm had a bug, but the results looked nice. Now we've "fixed" it
    # and the results look bad. #548 addresses this problem.
    @staticmethod
    def _richardson_lucy_deconv(
            image: np.ndarray, iterations: int, psf: np.ndarray, clip: bool) -> np.ndarray:
        """
        Deconvolves input image with a specified point spread function.

        Parameters
        ----------
        image : ndarray
           Input degraded image (can be N dimensional).
        psf : ndarray
           The point spread function.
        iterations : int
           Number of iterations. This parameter plays the role of
           regularisation.
        clip : boolean
            If true, pixel value of the result above 1 or
           under -1 are thresholded for skimage pipeline compatibility.

        Returns
        -------
        im_deconv : ndarray
           The deconvolved image.

        Examples
        --------
        >>> from skimage import color, data, restoration
        >>> camera = color.rgb2gray(data.camera())
        >>> from scipy.signal import convolve2d
        >>> psf = np.ones((5, 5)) / 25
        >>> camera = convolve2d(camera, psf, 'same')
        >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
        >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

        Notes
        -----
        This code is copied from skimage.restoration. We copied it to implement a bugfix wherein
        zeros in the input image or zeros produced during an intermediate would induce divide by
        zero -> Nan. These Nans would then propagate throughout the image, invalidating the results.
        Longer term, we will make a PR to skimage to introduce the fix. There is some existing work
        linked here: https://github.com/scikit-image/scikit-image/issues/2551

        """
        # compute the times for direct convolution and the fft method. The fft is of
        # complexity O(N log(N)) for each dimension and the direct method does
        # straight arithmetic (and is O(n*k) to add n elements k times)
        direct_time = np.prod(image.shape + psf.shape)
        fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

        # see whether the fourier transform convolution method or the direct
        # convolution method is faster (discussed in scikit-image PR #1792)
        time_ratio = 40.032 * fft_time / direct_time

        if time_ratio <= 1 or len(image.shape) > 2:
            convolve_method = fftconvolve
        else:
            convolve_method = convolve

        image = image.astype(np.float)
        psf = psf.astype(np.float)
        im_deconv = 0.5 * np.ones(image.shape)
        psf_mirror = psf[::-1, ::-1]

        eps = np.finfo(image.dtype).eps
        for _ in range(iterations):
            x = convolve_method(im_deconv, psf, 'same')
            np.place(x, x == 0, eps)
            relative_blur = image / x + eps
            im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

        if np.all(np.isnan(im_deconv)):
            raise RuntimeError(
                'All-NaN output data detected. Likely cause is that deconvolution has been run for '
                'too many iterations.')

        if clip:
            # Changing to cliping values above 1 here changes test results
            # so keeping this as rescaling for now
            im_deconv = preserve_float_range(im_deconv, True)

        return im_deconv

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose=False,
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
            if True, report on the percentage completed during processing (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        func = partial(
            self._richardson_lucy_deconv,
            iterations=self.num_iter, psf=self.psf, clip=self.clip
        )
        result = stack.apply(
            func,
            split_by={Indices.Y.value, Indices.X.value}, verbose=verbose, n_processes=n_processes,
            in_place=in_place
        )
        return result
