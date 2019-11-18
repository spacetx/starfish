from pathlib import Path
from typing import (
    Mapping,
    Optional,
    Sequence,
    Union,
)
from warnings import warn

from starfish.core.morphology.binary_mask.binary_mask import BinaryMaskCollection, MaskData
from starfish.core.morphology.label_image import label_image as li
from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from starfish.core.util.logging import Log


class SegmentationMaskCollection(BinaryMaskCollection):
    """Deprecated in favor of BinaryMaskCollection."""
    def __init__(
            self,
            pixel_ticks: Union[Mapping[Axes, ArrayLike[int]], Mapping[str, ArrayLike[int]]],
            physical_ticks: Union[Mapping[Coordinates, ArrayLike[Number]],
                                  Mapping[str, ArrayLike[Number]]],
            masks: Sequence[MaskData],
            log: Optional[Log],
    ):
        warn(
            f"{self.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        super().__init__(pixel_ticks, physical_ticks, masks, log)

    @classmethod
    def from_label_image(cls, label_image: li.LabelImage) -> "BinaryMaskCollection":
        warn(
            f"{cls.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        return BinaryMaskCollection.from_label_image(label_image)

    @classmethod
    def open_targz(cls, path: Union[str, Path]) -> "BinaryMaskCollection":
        warn(
            f"{cls.__class__.__name__} has been deprecated in favor of "
            f"{BinaryMaskCollection.__name__}",
            DeprecationWarning
        )
        return BinaryMaskCollection.open_targz(path)
