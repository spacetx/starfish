import itertools
from typing import List, Union, Tuple
import xarray as xr

from starfish.types import Features
from starfish.types._constants import OverlapStrategy


class Rect:
    def __init__(self, min_x, max_x, min_y, max_y):
        """
        Small class that defines a rectangle by its bottom left
        and top right coordinates.
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    @staticmethod
    def no_overlap(rect1: "Rect", rect2: "Rect") -> bool:
        """Return True if two rectangles do not overlap"""
        return (rect1.max_x < rect2.min_x) | \
               (rect1.min_x > rect2.max_x) | \
               (rect1.max_y < rect2.min_y) | \
               (rect1.min_y > rect2.max_y)

    @staticmethod
    def find_intersection(rect1: "Rect", rect2: "Rect") -> Union[None, "Rect"]:
        """
        Find the intersection area of two intensity tables and return as Rect object.
        If no overlap return none.
        """
        if Rect.no_overlap(rect1, rect2):
            return None
        return Rect(max(rect1.min_x, rect2.min_x),
                    min(rect1.max_x, rect2.max_x),
                    max(rect1.min_y, rect2.min_y),
                    min(rect1.max_y, rect2.max_y))


def find_overlaps_of_xarrays(xarrays: List[xr.DataArray]
                                  ) -> List[Tuple[int, int, "Rect"]]:
    """
    Takes a list of xarrays and returns a list of xarray pairs that overlap physically
    and their areas of overlap defined by a Rectangle.
    """
    all_overlaps:  List[Tuple[int, int, "Rect"]] = list()
    for idx1, idx2 in itertools.combinations(range(len(xarrays)), 2):
        xr1 = xarrays[idx1]
        xr2 = xarrays[idx2]
        rect1 = Rect(min(xr1['xc']).data,
                     max(xr1['xc']).data,
                     min(xr1['yc']).data,
                     max(xr1['yc']).data)

        rect2 = Rect(min(xr2['xc']).data,
                     max(xr2['xc']).data,
                     min(xr2['yc']).data,
                     max(xr2['yc']).data)
        intersection = Rect.find_intersection(rect1, rect2)
        if intersection:
            all_overlaps.append((idx1, idx2, intersection))
    return all_overlaps


def remove_area_of_xarray(it: xr.DataArray, area: Rect) -> xr.DataArray:
    """Remove section of xarray within a defined area"""
    return it.where((it.xc <= area.min_x) |
                    (it.xc >= area.max_x) |
                    (it.yc <= area.min_y) |
                    (it.yc >= area.max_y), drop=True)


def sel_area_of_xarray(it: xr.DataArray, area: Rect) -> xr.DataArray:
    """Select on xarray within a defined area"""
    return it.where((it.xc > area.min_x) &
                    (it.xc < area.max_x) &
                    (it.yc > area.min_y) &
                    (it.yc < area.max_y), drop=True)


def take_max(intersection_rect: Rect,
             it1: xr.DataArray,
             intersect1: xr.DataArray,
             it2: xr.DataArray,
             intersect2: xr.DataArray):
    # # compare to see which section has more spots
    if intersect1.sizes[Features.AXIS] >= intersect2.sizes[Features.AXIS]:
        # I1 has more spots remove intesection2 from it2
        it2 = remove_area_of_xarray(it2, intersection_rect)
    else:
        it1 = remove_area_of_xarray(it1, intersection_rect)
    return it1, it2


OVERLAP_STRATEGY_MAP = {
    OverlapStrategy.TAKE_MAX: take_max
}




