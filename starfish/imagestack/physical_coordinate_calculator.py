from typing import Mapping, Tuple, Union

import xarray as xr

from starfish.imagestack import indexing_utils
from starfish.types import Coordinates, Indices, PhysicalCoordinateTypes


def calc_new_physical_coords(image_stack, indexers) -> xr.DataArray:
    """Calculates the resulting coordinates array from indexing on each dimension in indexers

    Parameters
    ----------
    indexers : a dictionary of dim:index where index is the value
    or range to index the dimension

    """
    new_coords = image_stack._coordinates.copy()
    # index by R, CH, V
    key = {'r': indexers[Indices.ROUND.value],
           'c': indexers[Indices.CH.value],
           'z': indexers[Indices.Z.value]}
    new_coords = indexing_utils.index_keep_dimensions(new_coords, key)
    # check if X or Y dimension indexed, if so rescale
    if _needs_coords_recalculating(indexers[Indices.X.value], indexers[Indices.Y.value]):
        _recalculate_physical_coordinates(image_stack, indexers, new_coords)
    return new_coords


def _recalculate_physical_coordinates(image_stack, indexers: Mapping[str, Union[int, slice]],
                                      new_coords: xr.DataArray
                                      ) -> None:
    """Iterates through coordinates array and rescales x, y physical coordinate values
     based on how the x and y dimensions were indexed

    Parameters
    ----------
    indexers : a dictionary of dim:index where index is the value
    or range to index the dimension

    new_coords: the coordinates xarray to modify

    """
    for _round in range(new_coords.sizes[Indices.ROUND]):
        for ch in range(new_coords.sizes[Indices.CH]):
            for z in range(new_coords.sizes[Indices.Z]):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }
                xmin, xmax = _recalculate_physical_coordinate(
                    image_stack,
                    Coordinates.X,
                    indices,
                    indexers[Indices.X.value])
                ymin, ymax = _recalculate_physical_coordinate(
                    image_stack,
                    Coordinates.Y,
                    indices,
                    indexers[Indices.Y.value])

                new_coords[indices].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX]
                    )] = [xmin, xmax, ymin, ymax]


def _recalculate_physical_coordinate(image_stack, coord: Coordinates,
                                     tile_indices: Mapping[Indices, int],
                                     key) -> Tuple:
    """Calculates rescaled coordinates

    Parameters
    ----------
    coord: The coordinate to rescale.

    indices: The (Round, Ch, Z) indices that identify which tile's coordinates we're using

    key: a value or range to index on

    """
    coord_min, coord_max = image_stack.coordinates(tile_indices, coord)
    dimension_size = image_stack.xarray.sizes[coord.value[0]]
    physical_pixel_size = (coord_max - coord_min) / dimension_size
    new_min, new_max = coord_min, coord_max
    min_pixel_index = key if type(key) is int else key.start
    max_pixel_index = key if type(key) is int else key.stop
    if min_pixel_index:
        # check for negative index
        min_pixel_index = min_pixel_index + dimension_size if min_pixel_index < 0 \
            else min_pixel_index
        new_min = (physical_pixel_size * min_pixel_index) + coord_min
    if max_pixel_index:
        max_pixel_index = max_pixel_index + dimension_size if max_pixel_index < 0 \
            else max_pixel_index
        new_max = (physical_pixel_size * (max_pixel_index + 1)) + coord_min
    return new_min, new_max


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
