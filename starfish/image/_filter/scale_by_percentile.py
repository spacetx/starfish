from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Clip
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class ScaleByPercentile(FilterAlgorithmBase):

    def __init__(
        self, p: int=0, is_volume: bool=False,
        clip_method: Union[str, Clip]=Clip.CLIP
    ) -> None:
        """Image scaling filter

        Parameters
        ----------
        p : int
            each image in the stack is scaled by this percentile. must be in [0, 100]
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered. By default, filter 2-d (y, x) tiles
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
        self.p = p
        self.is_volume = is_volume
        self.clip_method = clip_method

    _DEFAULT_TESTING_PARAMETERS = {"p": 0}

    @staticmethod
    def _scale(image: Union[xr.DataArray, np.ndarray], p: int) -> np.ndarray:
        """Clip values of img below and above percentiles p_min and p_max

        Parameters
        ----------
        image : Union[xr.DataArray, np.ndarray
            image to be scaled
        p : int
            each image in the stack is scaled by this percentile. must be in [0, 100]

        Notes
        -----
        - Setting p to 100 scales the image by it's maximum value
        - No shifting or transformation to adjust dynamic range is done after scaling

        Returns
        -------
        np.ndarray :
          Numpy array of same shape as img

        """
        v = np.percentile(image, p)

        image = image / v

        return image

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=False,
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
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        clip = partial(self._scale, p=self.p)
        result = stack.apply(
            clip,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method
        )
        return result

    @staticmethod
    @click.command("ScaleByPercentile")
    @click.option(
        "--p", default=100, type=int, help="scale images by this percentile")
    @click.option(  # FIXME: was this intentionally missed?
        "--is-volume", is_flag=True, help="filter 3D volumes")
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, p, is_volume, clip_method):
        ctx.obj["component"]._cli_run(ctx, ScaleByPercentile(p, is_volume, clip_method))
