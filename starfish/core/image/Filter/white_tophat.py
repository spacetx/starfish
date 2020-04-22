from typing import Optional

import xarray as xr
from skimage.morphology import ball, disk, white_tophat

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels
from ._base import FilterAlgorithm
from .util import determine_axes_to_group_by


class WhiteTophat(FilterAlgorithm):
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

    Notes
    -----
    See `Top-hat transform`_ for more information

    .. _Top-hat transform: https://en.wikipedia.org/wiki/Top-hat_transform

    """

    def __init__(
            self,
            masking_radius: int,
            is_volume: bool = False,
            level_method: Levels = Levels.CLIP,
    ) -> None:
        self.masking_radius = masking_radius
        self.is_volume = is_volume
        self.level_method = level_method

    _DEFAULT_TESTING_PARAMETERS = {"masking_radius": 3}

    def _white_tophat(self, image: xr.DataArray) -> xr.DataArray:
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
    ) -> Optional[ImageStack]:
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
            level_method=self.level_method
        )
        return result
