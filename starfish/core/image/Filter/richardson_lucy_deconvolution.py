from functools import partial
from typing import Optional

import numpy as np
import xarray as xr
from scipy.signal import convolve, fftconvolve

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number
from ._base import FilterAlgorithm
from .util import (
    determine_axes_to_group_by,
    gaussian_kernel,
)


class DeconvolvePSF(FilterAlgorithm):
    """
    Deconvolve a point spread function from the image. The point spread function is assumed to be
    an isotropic Gaussian, with a user specified standard deviation, sigma.

    There are currently several issues with this function.
    See `issue #731 <https://github.com/spacetx/starfish/issues/731>`_

    Parameters
    ----------
    num_iter : int
        number of iterations to run. Note that this is a very important parameter that requires
        careful optimization
    sigma : Number
        standard deviation of the gaussian kernel used to construct the point spread function
    is_volume: bool
        If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
        independently.
    clip_method : Optional[Union[str, :py:class:`~starfish.types.Clip`]]
        Deprecated method to control the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Clip.CLIP: data above 1 are set to 1.  This has been replaced with
          level_method=Levels.CLIP.
        - Clip.SCALE_BY_IMAGE: when any data in the entire ImageStack is greater than 1, the entire
          ImageStack is scaled by the maximum value in the ImageStack.  This has been replaced with
          level_method=Levels.SCALE_SATURATED_BY_IMAGE.
        - Clip.SCALE_BY_CHUNK: when any data in any slice is greater than 1, each slice is scaled by
          the maximum value found in that slice.  The slice shapes are determined by the
          ``group_by`` parameters.  This has been replaced with
          level_method=Levels.SCALE_SATURATED_BY_CHUNK.
    level_method : :py:class:`~starfish.types.Levels`
        Controls the way that data are scaled to retain skimage dtype requirements that float data
        fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Levels.CLIP (default): data above 1 are set to 1.
        - Levels.SCALE_SATURATED_BY_IMAGE: when any data in the entire ImageStack is greater
          than 1, the entire ImageStack is scaled by the maximum value in the ImageStack.
        - Levels.SCALE_SATURATED_BY_CHUNK: when any data in any slice is greater than 1, each
          slice is scaled by the maximum value found in that slice.  The slice shapes are
          determined by the ``group_by`` parameters.
        - Levels.SCALE_BY_IMAGE: scale the entire ImageStack by the maximum value in the
          ImageStack.
        - Levels.SCALE_BY_CHUNK: scale each slice by the maximum value found in that slice.  The
          slice shapes are determined by the ``group_by`` parameters.

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
    #. `Richardson-Lucy Deconvolution`_

    .. _Richardson-Lucy Deconvolution: \
    https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

    Notes
    -----
    This code is based on code from :py:func:`skimage.restoration.richardson_lucy`. We modified it
    to implement a bugfix wherein zeros in the input image or zeros produced during an
    intermediate would induce divide by zero -> Nan. These Nans would then propagate throughout
    the image, invalidating the results. Longer term, we will make a PR to skimage to introduce
    the fix. There is some existing work linked here:
    `issue #2551 <https://github.com/scikit-image/scikit-image/issues/2551>`_

    """

    def __init__(
            self,
            num_iter: int,
            sigma: Number,
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP,
    ) -> None:

        self.num_iter = num_iter
        self.sigma = sigma
        self.kernel_size: int = int(2 * np.ceil(2 * sigma) + 1)
        self.psf: np.ndarray = gaussian_kernel(
            shape=(self.kernel_size, self.kernel_size),
            sigma=sigma
        )
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"num_iter": 2, "sigma": 1}

    # Here be dragons. This algorithm had a bug, but the results looked nice. Now we've "fixed" it
    # and the results look bad. #548 addresses this problem.
    @staticmethod
    def _richardson_lucy_deconv(
            image: xr.DataArray, iterations: int, psf: np.ndarray
    ) -> xr.DataArray:
        """
        Deconvolves input image with a specified point spread function.

        Parameters
        ----------
        image : xr.DataArray
           Input degraded image (can be N dimensional).
        psf : ndarray
           The point spread function.
        iterations : int
           Number of iterations. This parameter plays the role of
           regularisation.

        Returns
        -------
        im_deconv : xr.DataArray
           The deconvolved image.

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

        image = image.astype(float)
        psf = psf.astype(float)
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

        return xr.DataArray(im_deconv)

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose=False,
            n_processes: Optional[int] = None,
            *args,
    ) -> Optional[ImageStack]:
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
            Number of parallel processes to devote to applying the filter. If None, defaults to
            the result of os.cpu_count(). (default None)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        func = partial(
            self._richardson_lucy_deconv,
            iterations=self.num_iter, psf=self.psf
        )
        result = stack.apply(
            func,
            group_by=group_by,
            verbose=verbose,
            n_processes=n_processes,
            in_place=in_place,
            level_method=self.level_method,
        )
        return result
