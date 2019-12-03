from typing import Optional

import numpy as np
from scipy.ndimage import label

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from ._base import FilterAlgorithm


class StructuralLabel(FilterAlgorithm):
    """
    Using a BinaryMaskCollection with a single mask, produce a new BinaryMaskCollection with a mask
    representing each feature.  Features are determined using a structuring element, which determine
    what is considered part of a feature.

    Parameters
    ----------
    structure : Optional[np.ndarray]
        See the documentation for :py:func:`scipy.ndimage.label`
    """

    def __init__(self, structure: Optional[np.ndarray] = None) -> None:
        self._structure = structure

    def run(
            self,
            binary_mask_collection: BinaryMaskCollection,
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        assert len(binary_mask_collection) == 1
        mask = binary_mask_collection.uncropped_mask(0)
        labeled_array, _ = label(mask, self._structure)
        return BinaryMaskCollection.from_label_array_and_ticks(
            labeled_array,
            binary_mask_collection._pixel_ticks,
            binary_mask_collection._physical_ticks,
            binary_mask_collection.log,
        )
