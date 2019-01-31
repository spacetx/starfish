from typing import Any, Mapping, Optional, Tuple, Union

import xarray as xr

from starfish.imagestack import indexing_utils
from starfish.types import (
    Axes,
    Coordinates,
    Number,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes
)


def calc_new_physical_coords_array(
        physical_coordinates: xr.DataArray,
        stack_shape: Mapping[Axes, int],
        indexers: Mapping[str, Union[int, slice]],
) -> xr.DataArray:
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
    key = {Axes.ROUND.value: indexers[Axes.ROUND.value],
           Axes.CH.value: indexers[Axes.CH.value],
           Axes.ZPLANE.value: indexers[Axes.ZPLANE.value]}
    new_coords = indexing_utils.index_keep_dimensions(new_coords, key)
    # check if X or Y dimension indexed, if so recalculate physcal coordinate min/max values
    if _needs_coords_recalculating(indexers[Axes.X.value], indexers[Axes.Y.value]):
        _recalculate_physical_coordinate_ranges(stack_shape, indexers, new_coords)
    return new_coords


def _needs_coords_recalculating(
        x_indexers: Union[int, slice], y_indexers: Union[int, slice]) -> bool:
    if isinstance(x_indexers, int) or isinstance(y_indexers, int):
        return True
    return not (x_indexers.start is x_indexers.stop is y_indexers.start is y_indexers.stop is None)


def _recalculate_physical_coordinate_ranges(
        stack_shape: Mapping[Axes, int],
        indexers: Mapping[str, Union[int, slice]],
        coords_array: xr.DataArray,
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
    for _round in range(coords_array.sizes[Axes.ROUND]):
        for ch in range(coords_array.sizes[Axes.CH]):
            for z in range(coords_array.sizes[Axes.ZPLANE]):
                selector = {
                    Axes.ROUND: _round,
                    Axes.CH: ch,
                    Axes.ZPLANE: z
                }
                xmin, xmax, ymin, ymax = coords_array[selector].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX
                    ])]
                xmin, xmax = recalculate_physical_coordinate_range(
                    xmin, xmax,
                    stack_shape[Axes.X.value],
                    indexers[Axes.X.value])
                ymin, ymax = recalculate_physical_coordinate_range(
                    ymin, ymax,
                    stack_shape[Axes.Y.value],
                    indexers[Axes.Y.value])
                coords_array[selector].loc[
                    dict(physical_coordinate=[
                        PhysicalCoordinateTypes.X_MIN,
                        PhysicalCoordinateTypes.X_MAX,
                        PhysicalCoordinateTypes.Y_MIN,
                        PhysicalCoordinateTypes.Y_MAX
                    ])] = [xmin, xmax, ymin, ymax]


def _calculate_physical_pixel_size(coord_min: Number, coord_max: Number, num_pixels: int) -> Number:
    """Calculate the size of a pixel in physical space"""
    return (coord_max - coord_min) / num_pixels


def _pixel_offset_to_physical_coordinate(
        physical_pixel_size: Number,
        pixel_offset: Optional[int],
        coordinates_at_pixel_offset_0: Number,
        dimension_size: int,
) -> Number:
    """Calculate the physical pixel value at the given index"""
    if pixel_offset:
        # Check for negative index
        if pixel_offset < 0:
            pixel_offset = pixel_offset + dimension_size
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
    physical_pixel_size = _calculate_physical_pixel_size(coord_min, coord_max, dimension_size)
    min_pixel_index = indexer if isinstance(indexer, int) else indexer.start
    max_pixel_index = indexer if isinstance(indexer, int) else indexer.stop
    # Add one to max pixel index to get end of pixel
    max_pixel_index = max_pixel_index + 1 if max_pixel_index else dimension_size
    new_min = _pixel_offset_to_physical_coordinate(
        physical_pixel_size, min_pixel_index, coord_min, dimension_size)
    new_max = _pixel_offset_to_physical_coordinate(
        physical_pixel_size, max_pixel_index, coord_min, dimension_size)
    return new_min, new_max


def get_coordinates(
        coords_array: xr.DataArray,
        selector: Mapping[Axes, int],
        physical_axis: Coordinates,
) -> Tuple[float, float]:

    """Given a selector that uniquely identify a tile and a physical axis, return the min
    and the max coordinates for that tile along that axis."""

    selectors: Mapping[str, Any] = {
        Axes.ROUND.value: selector[Axes.ROUND],
        Axes.CH.value: selector[Axes.CH],
        Axes.ZPLANE.value: selector[Axes.ZPLANE],
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


def get_physical_coordinates_of_spot(
        coords_array: xr.DataArray,
        tile_selector: Mapping[Axes, int],
        pixel_x: int,
        pixel_y: int,
        tile_shape: Tuple[int, int]):
    """Given a selector that uniquely identify a tile and the location of a spot in pixel space,
    calculate the location in physical space."""
    x_range = get_coordinates(coords_array, tile_selector, Coordinates.X)
    physcial_pixel_size_x = _calculate_physical_pixel_size(coord_max=x_range[1],
                                                           coord_min=x_range[0],
                                                           num_pixels=tile_shape[1])
    physical_x = _pixel_offset_to_physical_coordinate(physical_pixel_size=physcial_pixel_size_x,
                                                      pixel_offset=pixel_x,
                                                      coordinates_at_pixel_offset_0=x_range[0],
                                                      dimension_size=tile_shape[1])

    y_range = get_coordinates(coords_array, tile_selector, Coordinates.Y)
    physcial_pixel_size_y = _calculate_physical_pixel_size(coord_max=y_range[1],
                                                           coord_min=y_range[0],
                                                           num_pixels=tile_shape[0])
    physical_y = _pixel_offset_to_physical_coordinate(physical_pixel_size=physcial_pixel_size_y,
                                                      pixel_offset=pixel_y,
                                                      coordinates_at_pixel_offset_0=y_range[0],
                                                      dimension_size=tile_shape[0])

    z_range = get_coordinates(coords_array, tile_selector, Coordinates.Z)
    # As discussed just taking the middle of the z range for this...unless we change our minds
    physical_z = (z_range[1] - z_range[0]) / 2 + z_range[0]

    return physical_x, physical_y, physical_z
