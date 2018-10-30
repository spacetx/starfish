from typing import Any, Mapping, Tuple, Union

import xarray as xr

from starfish.imagestack import indexing_utils
from starfish.types import (
    Coordinates,
    Indices,
    Number,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes
)


def calc_new_physical_coords_array(physical_coordinates: xr.DataArray,
                                   stack_shape: Mapping[Indices, int],
                                   indexers: Mapping[str, Union[int, slice]]) -> xr.DataArray:
    """Calculates the resulting coordinates array from indexing on each dimension in indexers

    Parameters
    ----------
    physical_coordinates: xarray
        xarray that holds the min/max values of physical coordinates per tile (Round, Ch, Z).
    stack_shape : Dict[str, int]
        The shape of the image tensor by categorical index (channels, imaging rounds, z-layers)
    indexers : Dict[str, (int/slice)]
        A dictionary of dim:index where index is the value or range to index the dimension

    Returns
    -------
    A coordinates xarray indexed by R, CH, V and values recalculated according to indexing on X/Y
    """
    new_coords = physical_coordinates.copy(deep=True)
    # index by R, CH, V
    key = {Indices.ROUND.value: indexers[Indices.ROUND.value],
           Indices.CH.value: indexers[Indices.CH.value],
           Indices.Z.value: indexers[Indices.Z.value]}
    new_coords = indexing_utils.index_keep_dimensions(new_coords, key)
    # check if X or Y dimension indexed, if so recalculate physcal coordinate min/max values
    if _needs_coords_recalculating(indexers[Indices.X.value], indexers[Indices.Y.value]):
        _recalculate_physical_coordinate_ranges(stack_shape, indexers, new_coords)
    return new_coords


def _needs_coords_recalculating(x_indexers: Union[int, slice], y_indexers: Union[int, slice]
                                ) -> bool:
    if isinstance(x_indexers, int) or isinstance(y_indexers, int):
        return True
    return not (x_indexers.start is x_indexers.stop is y_indexers.start is y_indexers.stop is None)


def _recalculate_physical_coordinate_ranges(stack_shape: Mapping[Indices, int],
                                            indexers: Mapping[str, Union[int, slice]],
                                            coords_array: xr.DataArray
                                            ) -> None:
    """Iterates through coordinates array and recalculates x, y min/max physical coordinate values
     based on how the x and y dimensions were indexed

    Parameters
    ----------
    indexers : Dict[str, (int/slice)]
        A dictionary of dim:index where index is the value or range to index the dimension

    coords_array: xarray
        xarray that holds the min/max values of physical coordinates per tile (Round, Ch, Z).

    Returns
    -------
    A coordinates xarray with values recalculated according to indexing on X/Y

    """
    for _round in range(coords_array.sizes[Indices.ROUND]):
        for ch in range(coords_array.sizes[Indices.CH]):
            for z in range(coords_array.sizes[Indices.Z]):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }
                xmin, xmax, ymin, ymax = coords_array[indices].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX
                    ])]
                xmin, xmax = _recalculate_physical_coordinate_range(
                    xmin, xmax,
                    stack_shape[Indices.X.value],
                    indexers[Indices.X.value])
                ymin, ymax = _recalculate_physical_coordinate_range(
                    ymin, ymax,
                    stack_shape[Indices.Y.value],
                    indexers[Indices.Y.value])
                coords_array[indices].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX
                    ])] = [xmin, xmax, ymin, ymax]


def _calculate_physcial_pixel_size(coord_max: Number, coord_min: Number, num_pixels: int
                                   ) -> Number:
    """Calculate the size of a pixel in physical space"""
    return (coord_max - coord_min) / num_pixels


def _pixel_offset_to_physical_coordinate(physical_pixel_size: Number,
                                         pixel_offset: float,
                                         coordinates_at_pixel_offset_0: Number,
                                         dimension_size: int
                                         ) -> Number:
    """Calculate the physical pixel value at the given index"""
    if pixel_offset:
        # Check for negative index
        if pixel_offset < 0:
            pixel_offset = pixel_offset + dimension_size
        return (physical_pixel_size * pixel_offset) + coordinates_at_pixel_offset_0
    return coordinates_at_pixel_offset_0


def _recalculate_physical_coordinate_range(coord_min: float,
                                           coord_max: float,
                                           dimension_size: int,
                                           indexer
                                           ) -> Tuple[Number, Number]:
    """Given the dimension size and pixel coordinate indexes calculate the corresponding
    coordinates in physical space

    Parameters
    ----------
    coord_min: float
        the minimum physical coordinate value

    coord_max: float
        the maximum physical coordinate value

    coord: Coordinate
        The (X, Y, or Z) coordinate to calculate.

    dimension_size: int
        The size (number of pixels) to use to calculate physical_pixel_size

    tile_indices: Dict[Indices, int]
        The (Round, Ch, Z) indices that identify which tile's coordinates we're using

    key: (int/slice)
        The pixel index or range to calculate.

    Returns
    -------
    The new min and max physical coordinate values of the given dimension
    """
    physical_pixel_size = _calculate_physcial_pixel_size(coord_max, coord_min, dimension_size)
    min_pixel_index = indexer if type(indexer) is int else indexer.start
    max_pixel_index = indexer if type(indexer) is int else indexer.stop
    # Add one to max pixel index to get end of pixel
    max_pixel_index = max_pixel_index + 1 if max_pixel_index else dimension_size
    new_min = _pixel_offset_to_physical_coordinate(physical_pixel_size, min_pixel_index,
                                                   coord_min, dimension_size)
    new_max = _pixel_offset_to_physical_coordinate(physical_pixel_size, max_pixel_index,
                                                   coord_min, dimension_size)
    return new_min, new_max


def get_coordinates(
        coords_array: xr.DataArray,
        indices: Mapping[Indices, int],
        physical_axis: Coordinates
) -> Tuple[float, float]:

    """Given a set of indices that uniquely identify a tile and a physical axis, return the min
    and the max coordinates for that tile along that axis."""

    selectors: Mapping[str, Any] = {
        Indices.ROUND.value: indices[Indices.ROUND],
        Indices.CH.value: indices[Indices.CH],
        Indices.Z.value: indices[Indices.Z],
    }
    min_selectors = dict(selectors)
    max_selectors = dict(selectors)
    if physical_axis == Coordinates.X:
        min_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.X_MIN
        max_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.X_MAX
    elif physical_axis == Coordinates.Y:
        min_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.Y_MIN
        max_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.Y_MAX
    elif physical_axis == Coordinates.Z:
        min_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.Z_MIN
        max_selectors[PHYSICAL_COORDINATE_DIMENSION] = PhysicalCoordinateTypes.Z_MAX

    return (
        coords_array.loc[min_selectors].item(),
        coords_array.loc[max_selectors].item(),
    )
