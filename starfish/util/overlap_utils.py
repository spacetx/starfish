import itertools
from typing import List, Union, Tuple
import xarray as xr

from starfish.types import Coordinates, Features
from starfish.types._constants import OverlapStrategy


class Area:
    def __init__(self, min_x, max_x, min_y, max_y):
        """
        Small class that defines an area of physical space by its bottom left
        and top right coordinates.
        """
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
        Find the intersection area of two areas and return as new Area object.
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
    Takes a list of xarrays and returns a list of xarray index pairs that overlap physically
    and their area of overlap defined by an Area object.
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
    """Remove area from xarray"""
    return it.where((it.xc <= area.min_x) |
                    (it.xc >= area.max_x) |
                    (it.yc <= area.min_y) |
                    (it.yc >= area.max_y), drop=True)


def sel_area_of_xarray(it: xr.DataArray, area: Area) -> xr.DataArray:
    """Select on xarray within a defined area"""
    return it.where((it.xc > area.min_x) &
                    (it.xc < area.max_x) &
                    (it.yc > area.min_y) &
                    (it.yc < area.max_y), drop=True)


def take_max(intersection_rect: Area,
             it1: xr.DataArray,
             it2: xr.DataArray
             ):
    # # compare to see which section has more spots
    intersect1 = sel_area_of_xarray(it1, intersection_rect)
    intersect2 = sel_area_of_xarray(it2, intersection_rect)
    if intersect1.sizes[Features.AXIS] >= intersect2.sizes[Features.AXIS]:
        # I1 has more spots remove intesection2 from it2
        it2 = remove_area_of_xarray(it2, intersection_rect)
    else:
        it1 = remove_area_of_xarray(it1, intersection_rect)
    return it1, it2


OVERLAP_STRATEGY_MAP = {
    OverlapStrategy.TAKE_MAX: take_max
}




