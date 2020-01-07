from typing import Mapping, Optional, Sequence

import numpy as np

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.util import _ticks_equal
from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from ._base import MergeAlgorithm


class SimpleMerge(MergeAlgorithm):
    """Merge multiple binary mask collections together.  This implementation requires that all
    the binary mask collections have the same pixel and physical ticks."""

    def run(
            self,
            binary_mask_collections: Sequence[BinaryMaskCollection],
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        """
        Parameters
        ----------
        binary_mask_collections : Sequence[BinaryMaskCollection]
            A sequence of binary mask collections with identical pixel and physical ticks.

        Returns
        -------
        BinaryMaskCollection
            A binary mask collection with the input mask collections merged together.
        """
        pixel_ticks: Optional[Mapping[Axes, ArrayLike[int]]] = None
        physical_ticks: Optional[Mapping[Coordinates, ArrayLike[Number]]] = None

        # validate that they have the same pixel/physical ticks.
        for binary_mask_collection in binary_mask_collections:
            pixel_ticks = pixel_ticks or binary_mask_collection._pixel_ticks
            physical_ticks = physical_ticks or binary_mask_collection._physical_ticks

            if not _ticks_equal(pixel_ticks, binary_mask_collection._pixel_ticks):
                raise ValueError("not all masks have the same pixel ticks")
            if not _ticks_equal(physical_ticks, binary_mask_collection._physical_ticks):
                raise ValueError("not all masks have the same physical ticks")

        # gather up all the uncropped masks.
        all_uncropped_masks = [
            np.asarray(binary_mask_collection.uncropped_mask(ix))
            for binary_mask_collection in binary_mask_collections
            for ix in range(len(binary_mask_collection))
        ]

        assert pixel_ticks is not None
        assert physical_ticks is not None

        return BinaryMaskCollection.from_binary_arrays_and_ticks(
            all_uncropped_masks,
            pixel_ticks,
            physical_ticks,
            None,
        )
