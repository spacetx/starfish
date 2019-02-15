"""Simple utility function we commonly use across ImageStack tests."""
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes, Coordinates, Number


def verify_stack_data(
        stack: ImageStack,
        selectors: Mapping[Axes, Union[int, slice]],
        expected_data: np.ndarray,
) -> Tuple[np.ndarray, Sequence[Axes]]:
    """Given an imagestack and a set of selectors, verify that the data referred to by the selectors
    matches the expected data.
    """
    tile_data, axes = stack.get_slice(selectors)
    assert np.array_equal(tile_data, expected_data)

    return tile_data, axes


def verify_stack_fill(
        stack: ImageStack,
        selectors: Mapping[Axes, Union[int, slice]],
        expected_fill_value: Number,
) -> Tuple[np.ndarray, Sequence[Axes]]:
    """Given an imagestack and a set of selectors, verify that the data referred to by the selectors
    matches an expected fill value.
    """
    tile_data, axes = stack.get_slice(selectors)
    expected_data = np.full(tile_data.shape, expected_fill_value, np.float32)
    assert np.array_equal(tile_data, expected_data)

    return tile_data, axes


def verify_physical_coordinates(
        stack: ImageStack,
        expected_x_coordinates: Tuple[float, float],
        expected_y_coordinates: Tuple[float, float],
        expected_z_coordinates: Tuple[float, float]
) -> None:
    """Given an imagestack and a set coordinate min/max values
    verify that the physical coordinates on the stack match the expected
    range of values for each coord dimension.
    """

    assert np.all(np.isclose(stack.xarray[Coordinates.X.value],
                             np.linspace(expected_x_coordinates[0],
                                         expected_x_coordinates[1],
                                         stack.xarray.sizes[Axes.X.value])))
    assert np.all(np.isclose(stack.xarray[Coordinates.Y.value],
                             np.linspace(expected_y_coordinates[0],
                                         expected_y_coordinates[1],
                                         stack.xarray.sizes[Axes.Y.value])))
    assert np.all(np.isclose(stack.xarray[Coordinates.Z.value], expected_z_coordinates))
