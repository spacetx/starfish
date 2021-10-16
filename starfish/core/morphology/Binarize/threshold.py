from typing import Mapping, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.util import _get_axes_names
from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from ._base import BinarizeAlgorithm


class ThresholdBinarize(BinarizeAlgorithm):
    """Binarizes an image using a threshold.  Pixels that exceed the threshold are considered True
    and all remaining pixels are considered False.

    The image being binarized must be an ImageStack with num_rounds == 1 and num_chs == 1.
    """
    def __init__(self, threshold: Number):
        self.threshold = threshold

    def _binarize(self, result: np.ndarray, tile_data: Union[np.ndarray, xr.DataArray]) -> None:
        result[:] = np.asarray(tile_data) >= self.threshold

    def run(self, image: ImageStack, *args, **kwargs) -> BinaryMaskCollection:
        if image.num_rounds != 1:
            raise ValueError(
                f"{ThresholdBinarize.__name__} given an image with more than one round "
                f"{image.num_rounds}")
        if image.num_chs != 1:
            raise ValueError(
                f"{ThresholdBinarize.__name__} given an image with more than one channel "
                f"{image.num_chs}")

        result_array = np.empty(
            shape=[
                image.shape[axis]
                for axis, _ in zip(*_get_axes_names(3))
            ],
            dtype=bool)

        self._binarize(result_array, image.xarray[0, 0])

        pixel_ticks: Mapping[Axes, ArrayLike[int]] = {
            Axes(axis): axis_data
            for axis, axis_data in image.xarray.coords.items()
            if axis in _get_axes_names(3)[0]
        }
        physical_ticks: Mapping[Coordinates, ArrayLike[Number]] = {
            Coordinates(coord): coord_data
            for coord, coord_data in image.xarray.coords.items()
            if coord in _get_axes_names(3)[1]
        }

        return BinaryMaskCollection.from_binary_arrays_and_ticks(
            (result_array,),
            pixel_ticks,
            physical_ticks,
            image.log,
        )
