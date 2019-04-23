from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr
from trackpy import bandpass

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip, Number
from starfish.core.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class Bandpass(FilterAlgorithmBase):
    """
    Convolve with a Gaussian to remove short-wavelength noise and subtract out long-wavelength
    variations, retaining features of intermediate scale. This implementation relies on
    scipy.ndimage.filters.gaussian_filter.

    This method is a thin wrapper around :code:`trackpy.Bandpass`:

    http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.bandpass.html

    Parameters
    ----------
    lshort : float
        filter frequencies below this value
    llong : int
        filter frequencies above this odd integer value
    threshold : float
        zero any spots below this intensity value after background subtraction (default 0)
    truncate : float
        truncate the gaussian kernel, used by the gaussian filter, at this many standard
        deviations (default 4)
    is_volume : bool
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
        self, lshort: Number, llong: int, threshold: Number = 0, truncate: Number = 4,
        is_volume: bool = False, clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:
        self.lshort = lshort
        self.llong = llong

        if threshold is None:
            raise ValueError("Threshold cannot be None, please pass a float or integer")

        self.threshold = threshold
        self.truncate = truncate
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"lshort": 1, "llong": 3, "threshold": 0.01}

    @staticmethod
    def _bandpass(
            image: Union[xr.DataArray, np.ndarray],
            lshort: Number, llong: int, threshold: Number, truncate: Number
    ) -> np.ndarray:
        """Apply a bandpass filter to remove noise and background variation

        Parameters
        ----------
        image : Union[xr.DataArray, np.ndarray]
        lshort : float
            filter frequencies below this value
        llong : int
            filter frequencies above this odd integer value
        threshold : float
            zero any spots below this intensity value after background subtraction (default 0)
        truncate : float
            truncate the gaussian kernel, used by the gaussian filter, at this many standard
            deviations

        Returns
        -------
        np.ndarray :
            bandpassed image

        """
        bandpassed = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed

    def run(
            self,
            stack: ImageStack,
            in_place: bool=False,
            verbose: bool=False,
            n_processes: Optional[int]=None,
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
        bandpass_ = partial(
            self._bandpass,
            lshort=self.lshort, llong=self.llong, threshold=self.threshold, truncate=self.truncate
        )

        group_by = determine_axes_to_group_by(self.is_volume)

        result = stack.apply(
            bandpass_,
            group_by=group_by,
            in_place=in_place,
            n_processes=n_processes,
            clip_method=self.clip_method,
            verbose=verbose,
        )
        return result

    @staticmethod
    @click.command("Bandpass")
    @click.option(
        "--lshort", type=float, help="filter signals below this frequency")
    @click.option(
        "--llong", type=int, help="filter signals above this frequency")
    @click.option(
        "--threshold", default=0, type=float, help="zero pixels below this intensity value")
    @click.option(
        "--truncate", default=4, type=float,
        help="truncate the filter at this many standard deviations")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, lshort, llong, threshold, truncate, clip_method):
        ctx.obj["component"]._cli_run(
            ctx,
            Bandpass(lshort, llong, threshold, truncate, clip_method)
        )
