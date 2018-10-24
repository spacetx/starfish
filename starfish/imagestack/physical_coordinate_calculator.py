from typing import Any, Mapping, Tuple, Union

import numpy as np
import xarray as xr

from starfish.imagestack import indexing_utils
from starfish.types import (
    Coordinates,
    Indices,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes
)


def calc_new_physical_coords_array(physical_coordinates: xr.DataArray,
                                   stack_shape: Mapping[Indices, int],
                                   indexers) -> xr.DataArray:
    """Calculates the resulting coordinates array from indexing on each dimension in indexers

    Parameters
    ----------
    physical_coordinates: xarray
        xarray that holds the min/max values of physical coordinates per tile (Round, Ch, Z).
    stack_shape : Dict[str, int]
        The shape of the image tensor by categorical index (channels, imaging rounds, z-layers)
    indexers : Dict[str, (int/slice)]
        A dictionary of dim:index where index is the value or range to index the dimension
    """
    new_coords = physical_coordinates.copy()
    # index by R, CH, V
    key = {Indices.ROUND.value: indexers[Indices.ROUND.value],
           Indices.CH.value: indexers[Indices.CH.value],
           Indices.Z.value: indexers[Indices.Z.value]}
    new_coords = indexing_utils.index_keep_dimensions(new_coords, key)
    # check if X or Y dimension indexed, if so recalculate physcal coordinate min/max values
    if _needs_coords_recalculating(indexers[Indices.X.value], indexers[Indices.Y.value]):
        _recalculate_physical_coordinate_ranges(stack_shape, indexers, new_coords)
    return new_coords


def _needs_coords_recalculating(x_indexers, y_indexers) -> bool:
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

    """
    for _round in range(coords_array.sizes[Indices.ROUND]):
        for ch in range(coords_array.sizes[Indices.CH]):
            for z in range(coords_array.sizes[Indices.Z]):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }
                xmin, xmax = _recalculate_physical_coordinate_range(
                    coords_array,
                    stack_shape[Indices.X.value],
                    Coordinates.X,
                    indices,
                    indexers[Indices.X.value])
                ymin, ymax = _recalculate_physical_coordinate_range(
                    coords_array,
                    stack_shape[Indices.Y.value],
                    Coordinates.Y,
                    indices,
                    indexers[Indices.Y.value])

                coords_array[indices].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX
                    ])] = [xmin, xmax, ymin, ymax]


def calculate_physcial_pixel_size(coord_max, coord_min, num_pixels):
    """Calculate the size of a pixel in physical space"""
    return (coord_max - coord_min) / num_pixels


def calculate_physical_pixel_value(physcial_pixel_size, index, start_of_range):
    """Calculate the physical pixel value at the given index"""
    if index:
        # Check for negative index
        if index < 0:
            index = index + start_of_range
        return (physcial_pixel_size * index) + start_of_range
    return start_of_range


def _recalculate_physical_coordinate_range(coords_array: xr.DataArray,
                                           dimension_size, coord: Coordinates,
                                           tile_indices: Mapping[Indices, int],
                                           key
                                           ) -> Tuple:
    """Given the dimension size and pixel coordinate indexes calculate the corresponding
    coordinates in physical space

    Parameters
    ----------
    coords_array: xarray
        xarray that holds the min/max values of physical coordinates per tile (Round, Ch, Z).

    coord: Coordinate
        The (X, Y, or Z) coordinate to calculate.

    dimension_size: int
        The size (number of pixels) to use to calculate physical_pixel_size

    tile_indices: Dict[Indices, int]
        The (Round, Ch, Z) indices that identify which tile's coordinates we're using

    key: (int/slice)
        The pixel index or range to calculate.

    """
    coord_min, coord_max = get_coordinates(coords_array, tile_indices, coord)
    physical_pixel_size = calculate_physcial_pixel_size(coord_max, coord_min, dimension_size)
    min_pixel_index = key if type(key) is int else key.start
    max_pixel_index = key if type(key) is int else key.stop
    # Add one to max pixel index to get end of pixel
    max_pixel_index = max_pixel_index + 1 if max_pixel_index else None
    new_min = calculate_physical_pixel_value(physical_pixel_size, min_pixel_index, coord_min)
    new_max = calculate_physical_pixel_value(physical_pixel_size, max_pixel_index, coord_min)
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

def _needs_coords_recalculating(x_indexers, y_indexers) -> bool:
    """
    Takes in a dict of dim:indexes and returns true if indexing on either x or y dimension

    Parameters
    ----------
    indexers : a dictionary of dim:index where index is the value
    or range to index the dimension
    """
    if isinstance(x_indexers, int) or isinstance(y_indexers, int):
        return True
    return not (x_indexers.start is x_indexers.stop is y_indexers.start is y_indexers.stop is None)


def transfer_physical_coords_to_intensity_table(image_stack, intensity_table):
    # TODO
    # Add three new coords to xarray (xc, yc, zc)
    intensity_table['xc'] = intensity_table.features * 0
    intensity_table['yc'] = intensity_table.features * 0
    intensity_table['zc'] = intensity_table.features * 0
    for ind, feature in intensity_table.groupby('features'):
        for ch, round in np.ndindex(feature.data.shape):
            # if non zero value set coords
            if feature[ch][round].data > 0:
                # get pixel coords of this tile
                pixel_x = feature[ch][round].coords.get(Coordinates.X).data
                pixel_y = feature[ch][round].coords.get(Coordinates.Y).data
                pixel_z = feature[ch][round].coords.get(Coordinates.Z).data
                tile_indices = {
                    Indices.ROUND.value: round,
                    Indices.CH.value: ch,
                    Indices.Z.value: pixel_z,
                }
                physical_coords = get_physcial_coordinates(tile_indices, pixel_x, pixel_y)
                intensity_table['xc'][ind] = physical_coords[0]
                intensity_table['yc'][ind] = physical_coords[1]
                intensity_table['zc'][ind] = physical_coords[2]
                break

