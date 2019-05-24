import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from starfish.core.types import Coordinates, Features, Number, OverlapStrategy


class Area:
    """
    Small class that defines rectangular area of physical space by
    its bottom left and top right coordinates.
    """
    def __init__(self, min_x: Number, max_x: Number, min_y: Number, max_y: Number):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def __eq__(self, other) -> bool:
        return (self.min_x == other.min_x
                and self.min_y == other.min_y
                and self.max_x == other.max_x
                and self.max_y == other.max_y)

    @staticmethod
    def _overlap(area1: "Area", area2: "Area") -> bool:
        """Return True if two rectangles overlap"""
        if (area1.max_x < area2.min_x) or (area1.min_x > area2.max_x):
            return False
        if (area1.max_y < area2.min_y) or (area1.min_y > area2.max_y):
            return False
        return True

    @staticmethod
    def find_intersection(area1: "Area", area2: "Area") -> Optional["Area"]:
        """
        Find the overlap area of two rectangles and return as new Area object.
        If the two rectangles do not overlap, return None.
        """
        if Area._overlap(area1, area2):
            return Area(max(area1.min_x, area2.min_x),
                        min(area1.max_x, area2.max_x),
                        max(area1.min_y, area2.min_y),
                        min(area1.max_y, area2.max_y))
        return None


def find_overlaps_of_xarrays(xarrays: Sequence[xr.DataArray]) -> Sequence[Tuple[int, int]]:
    """
    Find all the overlap areas within a list of xarrays.

    Parameters
    ----------
    xarrays : List[xr.DataArray]
        The list of xarrays to find overlaps in.

    Returns
    -------
    List[Tuple[int, int]] :
        A list of tuples containing the indices of two overlapping
        IntensityTables.

    """
    all_overlaps: List[Tuple[int, int]] = list()
    for idx1, idx2 in itertools.combinations(range(len(xarrays)), 2):
        xr1 = xarrays[idx1]
        xr2 = xarrays[idx2]
        area1 = Area(np.min(xr1[Coordinates.X.value]),
                     np.max(xr1[Coordinates.X.value]),
                     np.min(xr1[Coordinates.Y.value]),
                     np.max(xr1[Coordinates.Y.value]))

        area2 = Area(np.min(xr2[Coordinates.X.value]),
                     np.max(xr2[Coordinates.X.value]),
                     np.min(xr2[Coordinates.Y.value]),
                     np.max(xr2[Coordinates.Y.value]))
        if Area._overlap(area1, area2):
            all_overlaps.append((idx1, idx2))
    return all_overlaps


def remove_area_of_xarray(it: xr.DataArray, area: Area) -> xr.DataArray:
    """Return everything in the xarray defined OUTSIDE the input area
    including values on the boundary of the area defined.

    Parameters
    ----------
    it: xr.DataArray
        The xarray to modify
    area: Area
        The area to not include in the modified xarray

    Returns
    -------
    xr.DataArray :
        The xarray without the defined area.
    """
    return it.where((it.xc <= area.min_x)
                    | (it.xc >= area.max_x)
                    | (it.yc <= area.min_y)
                    | (it.yc >= area.max_y),
                    drop=True)


def sel_area_of_xarray(it: xr.DataArray, area: Area) -> xr.DataArray:
    """Return everything in the xarray defined WITHIN the input area
    including values on the boundary of the area defined.

    Parameters
    ----------
    it: xr.DataArray
        The xarray to modify
    area: Area
        The area to include in the modified xarray

    Returns
    -------
     xr.DataArray :
        The xarray within the defined area.
    """
    return it.where((it.xc >= area.min_x)
                    & (it.xc <= area.max_x)
                    & (it.yc >= area.min_y)
                    & (it.yc <= area.max_y), drop=True)


def take_max(it1: xr.DataArray, it2: xr.DataArray):
    """
    Compare two overlapping xarrays and remove spots from whichever has less in the
    overlapping section.

    Parameters
    ----------
    it1 : xr.DataArray
        The first overlapping xarray
    it2 : xr.DataArray
        The second overlapping xarray
    """
    area1 = Area(np.min(it1[Coordinates.X.value]),
                 np.max(it1[Coordinates.X.value]),
                 np.min(it1[Coordinates.Y.value]),
                 np.max(it1[Coordinates.Y.value]))

    area2 = Area(np.min(it2[Coordinates.X.value]),
                 np.max(it2[Coordinates.X.value]),
                 np.min(it2[Coordinates.Y.value]),
                 np.max(it2[Coordinates.Y.value]))
    intersection_rect = Area.find_intersection(area1, area2)
    if not intersection_rect:
        raise ValueError("The given xarrays do not overlap")
    intersect1 = sel_area_of_xarray(it1, intersection_rect)
    intersect2 = sel_area_of_xarray(it2, intersection_rect)
    # compare to see which section has more spots
    if intersect1.sizes[Features.AXIS] >= intersect2.sizes[Features.AXIS]:
        # I1 has more spots remove intesection2 from it2
        it2 = remove_area_of_xarray(it2, intersection_rect)
    else:
        it1 = remove_area_of_xarray(it1, intersection_rect)
    return it1, it2


"""
The mapping between OverlapStrategy type and the method to use for each.
"""
OVERLAP_STRATEGY_MAP = {
    OverlapStrategy.TAKE_MAX: take_max
}
