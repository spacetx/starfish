from typing import Optional, Union

import numpy as np
import xarray as xr
from skimage.morphology import ball, disk, white_tophat

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip
from starfish.core.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class WhiteTophat(FilterAlgorithmBase):
    """
    Performs "white top hat" filtering of an image to enhance spots. White top hat filtering
    finds spots that are both smaller and brighter than their surroundings by subtracting an
    estimate of the background produced by a binary opening of the image using a disk-shaped
    structuring element.

    Parameters
    ----------
    masking_radius : int
        radius of the morphological masking structure in pixels
    is_volume : int
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

    Notes
    -----
    See https://en.wikipedia.org/wiki/Top-hat_transform for more information

    """

    def __init__(
        self, masking_radius: int, is_volume: bool = False,
        clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:

        self.masking_radius = masking_radius
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"masking_radius": 3}

    def _white_tophat(self, image: Union[xr.DataArray, np.ndarray]) -> np.ndarray:
        if self.is_volume:
            structuring_element = ball(self.masking_radius)
        else:
            structuring_element = disk(self.masking_radius)
        return white_tophat(image, selem=structuring_element)

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
            If True, report on the percentage completed (default = False) during processing
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
        result = stack.apply(
            self._white_tophat,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method
        )
        return result

    @staticmethod
    @click.command("WhiteTophat")
    @click.option(
        "--masking-radius", default=15, type=int,
        help="diameter of morphological masking disk in pixels")
    @click.option(  # FIXME: was this intentionally missed?
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, masking_radius, is_volume, clip_method):
        ctx.obj["component"]._cli_run(ctx, WhiteTophat(masking_radius, is_volume, clip_method))
