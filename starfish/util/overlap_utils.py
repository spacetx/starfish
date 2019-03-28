import itertools
from typing import List, Union, Tuple
import xarray as xr

from starfish.types import Coordinates, Features
from starfish.types._constants import OverlapStrategy


class Area:
    """
    Small class that defines rectangular area of physical space by
    its bottom left and top right coordinates.
    """
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    @staticmethod
    def no_overlap(area1: "Area", area2: "Area") -> bool:
        """Return True if two rectangles do not overlap"""
        return (area1.max_x < area2.min_x) | \
               (area1.min_x > area2.max_x) | \
               (area1.max_y < area2.min_y) | \
               (area1.min_y > area2.max_y)

    @staticmethod
    def find_intersection(area1: "Area", area2: "Area") -> Union[None, "Area"]:
        """
        Find the overlap area of two rectangles and return as new Area object.
        If no overlap return none.
        """
        if Area.no_overlap(area1, area2):
            return None
        return Area(max(area1.min_x, area2.min_x),
                    min(area1.max_x, area2.max_x),
                    max(area1.min_y, area2.min_y),
                    min(area1.max_y, area2.max_y))


def find_overlaps_of_xarrays(xarrays: List[xr.DataArray]
                             ) -> List[Tuple[int, int, "Area"]]:
    """
    Find all the overlap areas within a list of xarrays.

    Parameters
    ----------
    xarrays : List[xr.DataArray]
        The list of xarrays to find overlaps in.

    Returns
    -------
    List[Tuple[int, int, "Area"]] :
        A list of tuples containing the indices of two overlapping
        IntensityTables and their Area of intersection.

    """
    all_overlaps:  List[Tuple[int, int, "Area"]] = list()
    for idx1, idx2 in itertools.combinations(range(len(xarrays)), 2):
        xr1 = xarrays[idx1]
        xr2 = xarrays[idx2]
        area1 = Area(min(xr1[Coordinates.X.value]).data,
                     max(xr1[Coordinates.X.value]).data,
                     min(xr1[Coordinates.Y.value]).data,
                     max(xr1[Coordinates.Y.value]).data)

        area2 = Area(min(xr2[Coordinates.X.value]).data,
                     max(xr2[Coordinates.X.value]).data,
                     min(xr2[Coordinates.Y.value]).data,
                     max(xr2[Coordinates.Y.value]).data)
        intersection = Area.find_intersection(area1, area2)
        if intersection:
            all_overlaps.append((idx1, idx2, intersection))
    return all_overlaps


def remove_area_of_xarray(it: xr.DataArray, area: Area) -> xr.DataArray:
    """Return everything in the xarray defined OUTSIDE the input area

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
    return it.where((it.xc <= area.min_x) |
                    (it.xc >= area.max_x) |
                    (it.yc <= area.min_y) |
                    (it.yc >= area.max_y), drop=True)


def sel_area_of_xarray(it: xr.DataArray, area: Area) -> xr.DataArray:
    """Return everything in the xarray defined WITHIN the input area

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
    return it.where((it.xc > area.min_x) &
                    (it.xc < area.max_x) &
                    (it.yc > area.min_y) &
                    (it.yc < area.max_y), drop=True)


def take_max(intersection_rect: Area,
             it1: xr.DataArray,
             it2: xr.DataArray
             ):
    """
    Compare two overlapping xarrays and remove spots from whichever
    has less int the overlapping section.

    Parameters
    ----------
    intersection_rect : Area
        The area of physical overalp between two xarrays
    it1 : xr.DataArray
        The firs overlapping xarray
    it2 : xr.DataArray
        The second overlapping xarray
    """
    # # compare to see which section has more spots
    intersect1 = sel_area_of_xarray(it1, intersection_rect)
    intersect2 = sel_area_of_xarray(it2, intersection_rect)
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




