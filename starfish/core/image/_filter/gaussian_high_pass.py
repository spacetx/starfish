from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from starfish.core.image._filter.gaussian_low_pass import GaussianLowPass
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip, Number
from starfish.core.util import click
from starfish.core.util.dtype import preserve_float_range
from ._base import FilterAlgorithmBase
from .util import (
    determine_axes_to_group_by,
    validate_and_broadcast_kernel_size,
)


class GaussianHighPass(FilterAlgorithmBase):
    """
    Applies a Gaussian high pass filter to the ImageStack. This is useful to remove cellular
    autofluorescence, which is typically low frequency.

    This filter works by subtracting a Gaussian low pass filter filtered version of the input image
    from the input image itself. The Gaussian low pass filter is defined in :py:func:`scipy.ndimage`
    and used by :py:func:`skimage.filters.gaussian`.

    Parameters
    ----------
    sigma : Union[Number, Tuple[Number]]
        standard deviation of gaussian kernel
    is_volume : bool
        If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles
        independently.
    clip_method : Union[str, Clip]
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack
        Clip.SCALE_BY_CHUNK: data above 1 are scaled by the maximum value, with the maximum
        value calculated over each slice, where slice shapes are determined by the group_by
        parameters
    """

    def __init__(
        self, sigma: Union[Number, Tuple[Number]], is_volume: bool = False,
        clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:

        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume)
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 3}

    @staticmethod
    def _high_pass(
            image: Union[xr.DataArray, np.ndarray],
            sigma: Union[Number, Tuple[Number]],
            rescale: bool = False
    ) -> Union[xr.DataArray, np.ndarray]:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        image : Union[xr.DataArray, np.ndarray]
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
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
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
            Number of parallel processes to devote to applying the filter. If None, defaults to
            the result of os.cpu_count(). (default None)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        high_pass: Callable = partial(self._high_pass, sigma=self.sigma)
        result = stack.apply(
            high_pass,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method
        )
        return result

    @staticmethod
    @click.command("GaussianHighPass")
    @click.option("--sigma", type=float, help="standard deviation of gaussian kernel")
    @click.option("--is-volume", is_flag=True,
                  help="indicates that the image stack should be filtered in 3d")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, sigma, is_volume, clip_method):
        ctx.obj["component"]._cli_run(ctx, GaussianHighPass(sigma, is_volume, clip_method))
