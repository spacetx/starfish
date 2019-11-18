from typing import List, Mapping, MutableMapping, Optional, Tuple, Union

from starfish.core.types import ArrayLike, Axes, Coordinates, Number


def _get_axes_names(ndim: int) -> Tuple[List[Axes], List[Coordinates]]:
    """Get needed axes and coordinates given the number of dimensions.  The axes and coordinates are
    returned in the order expected for binary masks.  For instance, the first axis/coordinate
    should be the first index into the mask.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    Returns
    -------
    axes : List[Axes]
        Axes.
    coords : List[Coordinates]
        Coordinates.
    """
    if ndim == 2:
        axes = [Axes.Y, Axes.X]
        coords = [Coordinates.Y, Coordinates.X]
    elif ndim == 3:
        axes = [Axes.ZPLANE, Axes.Y, Axes.X]
        coords = [Coordinates.Z, Coordinates.Y, Coordinates.X]
    else:
        raise TypeError('expected 2- or 3-D image')

    return axes, coords


def _normalize_pixel_ticks(
        pixel_ticks: Optional[Union[
            Mapping[Axes, ArrayLike[int]],
            Mapping[str, ArrayLike[int]]]],
) -> MutableMapping[Axes, ArrayLike[int]]:
    """Given pixel ticks in a mapping from an axis or a string representing an axis, return a
    mapping from an axis.  The mapping may also not be present (i.e., None), in which an empty
    dictionary is returned.
    """
    return {
        Axes(axis): axis_data
        for axis, axis_data in (pixel_ticks or {}).items()
    }


def _normalize_physical_ticks(
        physical_ticks: Union[
            Mapping[Coordinates, ArrayLike[Number]],
            Mapping[str, ArrayLike[Number]]],
) -> Mapping[Coordinates, ArrayLike[Number]]:
    """Given physical coordinate ticks in a mapping from a coordinate or a string representing a
    coordinate, return a mapping from a coordinate.
    """
    return {
        Coordinates(coord): coord_data
        for coord, coord_data in physical_ticks.items()
    }
