# Definition of the processing class
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_laplace

from starfish.core.image._filter._base import FilterAlgorithmBase
from starfish.core.image._filter.util import (
    determine_axes_to_group_by,
    validate_and_broadcast_kernel_size,
)
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip, Number
from starfish.core.util import click


class Laplace(FilterAlgorithmBase):
    """
    Multi-dimensional Gaussian-Laplacian filter used to enhance dots against background

    This filter wraps :py:func:`scipy.ndimage.gaussian_laplace`

    Parameters
    ----------
    sigma : Union[Number, Tuple[Number]]
        Standard deviation for Gaussian kernel to enhance dots.
    mode : str
        The mode parameter determines how the input array is extended when
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
        (Default 0) Value to fill past edges of input if mode is ‘constant’.
    is_volume: bool
        If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) planes
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
        self,
        sigma: Union[Number, Tuple[Number]], mode: str = 'reflect',
        cval: float = 0.0, is_volume: bool = False, clip_method: Union[str, Clip] = Clip.CLIP,
    ) -> None:

        self.sigma = validate_and_broadcast_kernel_size(sigma, is_volume=is_volume)
        self.mode = mode
        self.cval = cval
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"sigma": 0.5}

    @staticmethod
    def _gaussian_laplace(
        image: Union[xr.DataArray, np.ndarray], sigma: Union[Number, Tuple[Number]],
        mode: str = 'reflect', cval: float = 0.0
    ) -> np.ndarray:
        filtered = gaussian_laplace(
            image, sigma=sigma, mode=mode, cval=cval)

        filtered = -filtered  # the peaks are negative so invert the signal

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
        apply_filtering: Callable = partial(self._gaussian_laplace, sigma=self.sigma)
        return stack.apply(
            apply_filtering,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method
        )

    @staticmethod
    @click.command("Laplace")
    @click.option(
        "--sigma", type=float,
        help="Standard deviation of gaussian kernel for spot enhancement")
    @click.option(
        "--mode", default="reflect",
        help="How the input array is extended when the filter overlaps a border")
    @click.option(
        "--cval", default=0.0,
        help="Value to fill past edges of input if mode is ‘constant")
    @click.option(
        "--is-volume", is_flag=True,
        help="indicates that the image stack should be filtered in 3d")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, sigma, mode, cval, is_volume, clip_method):
        ctx.obj["component"]._cli_run(ctx, Laplace(sigma, mode, cval, is_volume, clip_method))
