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
        selectors: Mapping[Axes, int],
        expected_x_coordinates: Tuple[float, float],
        expected_y_coordinates: Tuple[float, float],
        expected_z_coordinates: Tuple[float, float]
) -> None:
    """Given an imagestack and a set of selectors, verify that the physical coordinates for the data
    referred to by the selectors match the expected physical coordinates.
    """
    assert np.all(np.isclose(
        stack.tile_coordinates(selectors, Coordinates.X),
        expected_x_coordinates))
    assert np.all(np.isclose(
        stack.tile_coordinates(selectors, Coordinates.Y),
        expected_y_coordinates))
    assert np.all(np.isclose(
        stack.tile_coordinates(selectors, Coordinates.Z),
        expected_z_coordinates))
