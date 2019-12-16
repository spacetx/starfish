import numpy as np
import pytest

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.binary_mask.test.factories import (
    binary_arrays_2d,
    binary_mask_collection_2d,
    binary_mask_collection_3d,
)
from starfish.core.morphology.util import _ticks_equal
from starfish.core.types import Axes, Coordinates
from ..simple import SimpleMerge


def test_success():
    mask_collection_0 = binary_mask_collection_2d()
    binary_arrays, physical_ticks = binary_arrays_2d()
    binary_arrays_negated = [
        np.bitwise_not(binary_array)
        for binary_array in binary_arrays
    ]
    mask_collection_1 = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays_negated, None, physical_ticks, None)

    merged = SimpleMerge().run([mask_collection_0, mask_collection_1])

    assert _ticks_equal(merged._pixel_ticks, mask_collection_0._pixel_ticks)
    assert _ticks_equal(merged._physical_ticks, mask_collection_0._physical_ticks)
    assert len(mask_collection_0) + len(mask_collection_1) == len(merged)

    # go through all the original uncroppped masks, and verify that they are somewhere in the merged
    # set.
    for mask_collection in (mask_collection_0, mask_collection_1):
        for ix in range(len(mask_collection)):
            uncropped_original_mask = mask_collection.uncropped_mask(ix)
            for jx in range(len(merged)):
                uncropped_copy_mask = merged.uncropped_mask(jx)

                if uncropped_original_mask.equals(uncropped_copy_mask):
                    # found the copy, break
                    break
            else:
                pytest.fail("could not find mask in merged set.")

def test_pixel_tick_mismatch():
    mask_collection_0 = binary_mask_collection_2d()
    mask_collection_0._pixel_ticks[Axes.X.value] = np.asarray(
        mask_collection_0._pixel_ticks[Axes.X.value]) + 1
    binary_arrays, physical_ticks = binary_arrays_2d()
    binary_arrays_negated = [
        np.bitwise_not(binary_array)
        for binary_array in binary_arrays
    ]
    mask_collection_1 = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays_negated, None, physical_ticks, None)

    with pytest.raises(ValueError):
        SimpleMerge().run([mask_collection_0, mask_collection_1])


def test_physical_tick_mismatch():
    mask_collection_0 = binary_mask_collection_2d()
    mask_collection_0._physical_ticks[Coordinates.X] = np.asarray(
        mask_collection_0._physical_ticks[Coordinates.X]) + 1
    binary_arrays, physical_ticks = binary_arrays_2d()
    binary_arrays_negated = [
        np.bitwise_not(binary_array)
        for binary_array in binary_arrays
    ]
    mask_collection_1 = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays_negated, None, physical_ticks, None)

    with pytest.raises(ValueError):
        SimpleMerge().run([mask_collection_0, mask_collection_1])


def test_shape_mismatch():
    mask_collection_0 = binary_mask_collection_2d()
    mask_collection_1 = binary_mask_collection_3d()

    with pytest.raises(ValueError):
        SimpleMerge().run([mask_collection_0, mask_collection_1])
