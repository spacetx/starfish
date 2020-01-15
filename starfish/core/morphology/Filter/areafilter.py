import warnings
from typing import MutableSequence, Optional

from starfish.core.morphology.binary_mask import BinaryMaskCollection, MaskData
from ._base import FilterAlgorithm


class AreaFilter(FilterAlgorithm):
    """
    Create a BinaryMaskCollection using only the masks that meet the minimum and maximum area
    criteria.

    Parameters
    ----------
    min_area : Optional[int]
        Only masks that have area larger or equal to this will be included in the output mask
        collection.
    max_area : Optional[int]
        Only masks that have area smaller or equal to this will be included in the output mask
        collection.
    """

    def __init__(
            self,
            min_area_DEPRECATED: Optional[int] = None,
            max_area_DEPRECATED: Optional[int] = None,
            *,
            min_area: Optional[int] = None,
            max_area: Optional[int] = None,
    ):
        if min_area_DEPRECATED is not None:
            if min_area is not None:
                raise ValueError(
                    "Cannot specify both min_area as a positional argument and a keyword "
                    "argument.  AreaFilter should be initialized only by "
                    "`AreaFilter(min_area=xyz)`.")
            warnings.warn(
                "min_area should be provided as a keyword argument, i.e., "
                "`AreaFilter(min_area=xyz)`.")
            min_area = min_area_DEPRECATED
        if max_area_DEPRECATED is not None:
            if max_area is not None:
                raise ValueError(
                    "Cannot specify both max_area as a positional argument and a keyword "
                    "argument.  AreaFilter should be initialized only by "
                    "`AreaFilter(max_area=xyz)`.")
            warnings.warn(
                "max_area should be provided as a keyword argument, i.e., "
                "`AreaFilter(max_area=xyz)`.")
            max_area = max_area_DEPRECATED

        self._min_area = min_area
        self._max_area = max_area

        if (self._min_area is not None
                and self._max_area is not None
                and self._min_area > self._max_area):
            raise ValueError(f"min_area ({min_area}) should be smaller than max_area ({max_area})")

    def run(
            self,
            binary_mask_collection: BinaryMaskCollection,
            *args, **kwargs) -> BinaryMaskCollection:
        matching_mask_data: MutableSequence[MaskData] = list()
        for ix in range(len(binary_mask_collection)):
            props = binary_mask_collection.mask_regionprops(ix)
            if self._min_area is not None and props.area < self._min_area:
                continue
            if self._max_area is not None and props.area > self._max_area:
                continue

            matching_mask_data.append(binary_mask_collection._masks[ix])

        return BinaryMaskCollection(
            binary_mask_collection._pixel_ticks,
            binary_mask_collection._physical_ticks,
            matching_mask_data,
            binary_mask_collection._log,
        )
