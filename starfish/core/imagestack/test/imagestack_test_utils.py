"""Simple utility function we commonly use across ImageStack tests."""
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from starfish.core.types import Axes, Coordinates, Number
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


def verify_physical_coordinates(
        stack: ImageStack,
        expected_x_coordinates: Tuple[Number, Number],
        expected_y_coordinates: Tuple[Number, Number],
        expected_z_coordinates: Union[Number, Tuple[Number, Number]],
        zplane: Optional[int] = None,
) -> None:
    """Given an imagestack and a set of coordinate min/max values verify that the physical
    coordinates on the stack match the expected range of values for each coord dimension.
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
        assert np.isclose(
            stack.xarray.sel({Axes.ZPLANE.value: zplane})[Coordinates.Z.value],
            expected_z_coordinates)
    else:
        assert np.all(np.isclose(stack.xarray[Coordinates.Z.value], expected_z_coordinates))


def _calculate_physical_pixel_size(coord_min: Number, coord_max: Number, num_pixels: int) -> Number:
    """Calculate the size of a pixel in physical space"""
    return (coord_max - coord_min) / num_pixels


def _pixel_offset_to_physical_coordinate(
        physical_pixel_size: Number,
        pixel_offset: Optional[int],
        coordinates_at_pixel_offset_0: Number,
) -> Number:
    """Calculate the physical pixel value at the given index"""
    if pixel_offset:
        # Check for negative index
        assert pixel_offset >= 0
        return (physical_pixel_size * pixel_offset) + coordinates_at_pixel_offset_0
    return coordinates_at_pixel_offset_0


def recalculate_physical_coordinate_range(
        coord_min: float,
        coord_max: float,
        dimension_size: int,
        indexer: Union[int, slice],
) -> Tuple[Number, Number]:
    """Given the dimension size and pixel coordinate indexes calculate the corresponding
    coordinates in physical space

    Parameters
    ----------
    coord_min: float
        the minimum physical coordinate value

    coord_max: float
        the maximum physical coordinate value

    dimension_size: int
        The size (number of pixels) to use to calculate physical_pixel_size

    indexer: (int/slice)
        The pixel index or range to calculate.

    Returns
    -------
    The new min and max physical coordinate values of the given dimension
    """
    physical_pixel_size = _calculate_physical_pixel_size(coord_min, coord_max, dimension_size - 1)
    min_pixel_index = indexer if isinstance(indexer, int) else indexer.start
    if isinstance(indexer, int):
        max_pixel_index = indexer
    elif isinstance(indexer.stop, int):
        if indexer.stop >= 0:
            max_pixel_index = indexer.stop - 1
        else:
            max_pixel_index = indexer.stop + dimension_size
    else:
        max_pixel_index = dimension_size - 1
    new_min = _pixel_offset_to_physical_coordinate(physical_pixel_size, min_pixel_index, coord_min)
    new_max = _pixel_offset_to_physical_coordinate(physical_pixel_size, max_pixel_index, coord_min)
    return new_min, new_max
