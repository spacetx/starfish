import numpy as np
import pytest

from starfish.core.types import Axes, Coordinates
from .factories import binary_arrays_2d, binary_arrays_3d
from ..binary_mask import BinaryMaskCollection


def test_2d():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with 2D data.  Pixel ticks
    are inferred."""
    binary_arrays, physical_ticks = binary_arrays_2d()

    binary_mask_collection = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays,
        None,
        physical_ticks,
        None
    )
    assert len(binary_mask_collection) == 2

    region_0, region_1 = binary_mask_collection.masks()
    assert region_0.name == '0'
    assert region_1.name == '1'

    assert np.array_equal(region_0, np.ones((1, 6), dtype=np.bool))
    temp = np.ones((2, 3), dtype=np.bool)
    temp[-1, -1] = False
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.Y.value], [0])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4, 5])

    assert np.array_equal(region_1[Axes.Y.value], [3, 4])
    assert np.array_equal(region_1[Axes.X.value], [3, 4, 5])

    assert np.array_equal(region_0[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][0:1])
    assert np.array_equal(region_0[Coordinates.X.value],
                          physical_ticks[Coordinates.X][0:6])

    assert np.array_equal(region_1[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][3:6])
    assert np.array_equal(region_1[Coordinates.X.value],
                          physical_ticks[Coordinates.X][3:6])

    # verify that we can lazy-calculate the regionprops correctly.
    region_0_props = binary_mask_collection.mask_regionprops(0)
    region_1_props = binary_mask_collection.mask_regionprops(1)
    assert region_0_props.area == 6
    assert region_1_props.area == 5


def test_3d():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with 3D data.  Pixel ticks
    are inferred."""
    binary_arrays, physical_ticks = binary_arrays_3d()

    binary_mask_collection = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays,
        None,
        physical_ticks,
        None
    )
    assert len(binary_mask_collection) == 2

    region_0, region_1 = binary_mask_collection.masks()
    assert region_0.name == '0'
    assert region_1.name == '1'

    assert np.array_equal(region_0, np.ones((1, 1, 6), dtype=np.bool))
    temp = np.ones((2, 2, 3), dtype=np.bool)
    temp[-1, -1, -1] = False
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.ZPLANE.value], [0])
    assert np.array_equal(region_0[Axes.Y.value], [0])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4, 5])

    assert np.array_equal(region_1[Axes.ZPLANE.value], [0, 1])
    assert np.array_equal(region_1[Axes.Y.value], [3, 4])
    assert np.array_equal(region_1[Axes.X.value], [3, 4, 5])

    assert np.array_equal(region_0[Coordinates.Z.value],
                          physical_ticks[Coordinates.Z][0:1])
    assert np.array_equal(region_0[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][0:1])
    assert np.array_equal(region_0[Coordinates.X.value],
                          physical_ticks[Coordinates.X][0:6])

    assert np.array_equal(region_1[Coordinates.Z.value],
                          physical_ticks[Coordinates.Z][0:2])
    assert np.array_equal(region_1[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][3:5])
    assert np.array_equal(region_1[Coordinates.X.value],
                          physical_ticks[Coordinates.X][3:6])

    # verify that we can lazy-calculate the regionprops correctly.
    region_0_props = binary_mask_collection.mask_regionprops(0)
    region_1_props = binary_mask_collection.mask_regionprops(1)
    assert region_0_props.area == 6
    assert region_1_props.area == 11


def test_no_mask():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with no masks.  Pixel ticks
    are inferred."""
    _, physical_ticks = binary_arrays_2d()

    binary_mask_collection = BinaryMaskCollection.from_binary_arrays_and_ticks(
        [],
        None,
        physical_ticks,
        None
    )
    assert len(binary_mask_collection) == 0

    assert np.array_equal(binary_mask_collection._pixel_ticks[Axes.X], np.arange(0, 6))
    assert np.array_equal(binary_mask_collection._pixel_ticks[Axes.Y], np.arange(0, 5))
    assert np.array_equal(
        binary_mask_collection._physical_ticks[Coordinates.X],
        physical_ticks[Coordinates.X])
    assert np.array_equal(
        binary_mask_collection._physical_ticks[Coordinates.Y],
        physical_ticks[Coordinates.Y])


def test_empty_mask():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with no masks.  Pixel ticks
    are inferred."""
    binary_arrays = [
        np.zeros((5, 6), dtype=np.bool),
    ]
    _, physical_ticks = binary_arrays_2d()

    binary_mask_collection = BinaryMaskCollection.from_binary_arrays_and_ticks(
        binary_arrays,
        None,
        physical_ticks,
        None
    )
    assert len(binary_mask_collection) == 1

    assert binary_mask_collection[0].shape == (0, 0)

    assert np.array_equal(binary_mask_collection._pixel_ticks[Axes.X], np.arange(0, 6))
    assert np.array_equal(binary_mask_collection._pixel_ticks[Axes.Y], np.arange(0, 5))
    assert np.array_equal(
        binary_mask_collection._physical_ticks[Coordinates.X],
        physical_ticks[Coordinates.X])
    assert np.array_equal(
        binary_mask_collection._physical_ticks[Coordinates.Y],
        physical_ticks[Coordinates.Y])


def test_mismatched_binary_array_sizes():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with 2D data.  Not all
    arrays are sized identically."""
    binary_arrays = [
        np.zeros((3, 6), dtype=np.bool),
        np.zeros((5, 6), dtype=np.bool),
    ]
    binary_arrays[0][0] = True
    binary_arrays[1][3:5, 3:6] = True
    binary_arrays[1][-1, -1] = False

    _, physical_ticks = binary_arrays_2d()

    with pytest.raises(ValueError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_arrays,
            None,
            physical_ticks,
            None
        )


def test_mismatched_binary_array_types():
    """Simple case of BinaryMaskCollection.from_binary_arrays_and_ticks with 2D data.  Not all
    arrays are of the correct type."""
    binary_arrays = [
        np.zeros((3, 6), dtype=np.bool),
        np.zeros((3, 6), dtype=np.int),
    ]
    binary_arrays[0][0] = True
    binary_arrays[1][3:5, 3:6] = True
    binary_arrays[1][-1, -1] = False

    _, physical_ticks = binary_arrays_2d()

    with pytest.raises(TypeError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_arrays,
            None,
            physical_ticks,
            None
        )


def test_incorrectly_sized_pixel_ticks():
    """BinaryMaskCollection.from_binary_arrays_and_ticks with 2D data.  Pixel ticks are incorrectly
    sized."""
    binary_arrays, physical_ticks = binary_arrays_2d()

    pixel_ticks = {
        Axes.Y: [0, 1, 2, 3],
        Axes.X: [0, 1, 2, 3, 4, 5],
    }

    with pytest.raises(ValueError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_arrays,
            pixel_ticks,
            physical_ticks,
            None
        )


def test_missing_physical_ticks():
    """BinaryMaskCollection.from_binary_arrays_and_ticks with some physical ticks missing."""
    binary_arrays = [
        np.zeros((2, 5, 6), dtype=np.bool),
        np.zeros((2, 5, 6), dtype=np.bool),
    ]

    physical_ticks = {
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]}

    # Z physical ticks missing
    with pytest.raises(ValueError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_arrays,
            None,
            physical_ticks,
            None
        )


def test_incorrectly_sized_physical_ticks():
    """BinaryMaskCollection.from_label_array_and_ticks with some physical ticks incorrectly
    sized."""
    binary_2d_arrays = [
        np.zeros((5, 6), dtype=np.bool),
        np.zeros((5, 6), dtype=np.bool),
    ]

    physical_ticks_2d = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                         Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12]}

    with pytest.raises(ValueError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_2d_arrays,
            None,
            physical_ticks_2d,
            None
        )

    binary_3d_arrays = [
        np.zeros((2, 5, 6), dtype=np.bool),
        np.zeros((2, 5, 6), dtype=np.bool),
    ]

    physical_ticks_3d = {
        Coordinates.Z: [0.0, 1.0],
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]
    }

    with pytest.raises(ValueError):
        BinaryMaskCollection.from_binary_arrays_and_ticks(
            binary_3d_arrays,
            None,
            physical_ticks_3d,
            None
        )
