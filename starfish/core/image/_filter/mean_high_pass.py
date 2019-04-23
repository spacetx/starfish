from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage.filters import uniform_filter

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip, Number
from starfish.core.util import click
from starfish.core.util.dtype import preserve_float_range
from ._base import FilterAlgorithmBase
from .util import (
    determine_axes_to_group_by, validate_and_broadcast_kernel_size
)


class MeanHighPass(FilterAlgorithmBase):
    """
    The mean high pass filter reduces low spatial frequency features by subtracting a
    mean filtered image from the original image. The mean filter smooths an image by replacing
    each pixel's value with an average of the pixel values of the surrounding neighborhood.

    The mean filter is also known as a uniform or box filter. It can also be considered as a fast
    approximation to a GaussianHighPass filter.

    This is a pass through for the scipy.ndimage.filters.uniform_filter:
    https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.ndimage.uniform_filter.html

    Parameters
    ----------
    size : Union[Number, Tuple[Number]]
        width of the kernel
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
        self, size: Union[Number, Tuple[Number]], is_volume: bool = False,
        clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:

        self.size = validate_and_broadcast_kernel_size(size, is_volume)
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"size": 1}

    @staticmethod
    def _high_pass(
        image: Union[xr.DataArray, np.ndarray], size: Number, rescale: bool = False
    ) -> np.ndarray:
        """
        Applies a mean high pass filter to an image

        Parameters
        ----------
        image : Union[xr.DataArray, numpy.ndarray]
            2-d or 3-d image data
        size : Union[Number, Tuple[Number]]
            width of the kernel
        rescale : bool
            If true scales data by max value, if false clips max values to one

        Returns
        -------
        np.ndarray[np.float32]:
            Filtered image, same shape as input
        """

        blurred: np.ndarray = uniform_filter(image, size)

        filtered: np.ndarray = image - blurred
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
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        high_pass: Callable = partial(self._high_pass, size=self.size)
        result = stack.apply(
            high_pass,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method
        )
        return result

    @staticmethod
    @click.command("MeanHighPass")
    @click.option(
        "--size", type=float, help="width of the kernel")
    @click.option(
        "--is-volume", is_flag=True,
        help="indicates that the image stack should be filtered in 3d")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, size, is_volume, clip_method):
        ctx.obj["component"]._cli_run(ctx, MeanHighPass(size, is_volume, clip_method))
