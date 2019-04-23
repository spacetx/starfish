"""Simple utility function we commonly use across ImageStack tests."""
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from starfish.core.types import Axes, Coordinates
from ..imagestack import ImageStack


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


def verify_physical_coordinates(stack: ImageStack,
                                expected_x_coordinates: Tuple[float, float],
                                expected_y_coordinates: Tuple[float, float],
                                expected_z_coordinates: Tuple[float, float],
                                zplane: Optional[int] = None) -> None:
    """Given an imagestack and a set of coordinate min/max values
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
    # If zplane provided, test expected_z_coordinates on specific plane.
    # Else just test expected_z_coordinates on entire array
    if zplane is not None:
        assert np.isclose(stack.xarray[Coordinates.Z.value][zplane], expected_z_coordinates)
    else:
        assert np.all(np.isclose(stack.xarray[Coordinates.Z.value], expected_z_coordinates))
