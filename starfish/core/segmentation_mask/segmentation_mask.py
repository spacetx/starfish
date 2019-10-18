from typing import (
    Dict,
    Iterable,
    Optional,
    Sequence,
)
from warnings import warn

import numpy as np
import xarray as xr
from skimage.measure._regionprops import _RegionProperties

from starfish.core.binary_mask import BinaryMaskCollection
from starfish.core.types import Coordinates


class SegmentationMaskCollection(BinaryMaskCollection):
    """Deprecated in favor of BinaryMaskCollection."""
    def __init__(
            self,
            masks: Iterable[xr.DataArray],
            props: Optional[Iterable[Optional[_RegionProperties]]] = None,
    ):
        warn(
            f"{self.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        super().__init__(masks, props)

    @classmethod
    def from_label_image(
            cls,
            label_image: np.ndarray,
            physical_ticks: Dict[Coordinates, Sequence[float]]
    ) -> "BinaryMaskCollection":
        warn(
            f"{cls.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        return BinaryMaskCollection.from_label_image(label_image, physical_ticks)

    @classmethod
    def from_disk(cls, path: str) -> "BinaryMaskCollection":
        warn(
            f"{cls.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        return BinaryMaskCollection.from_disk(path)
