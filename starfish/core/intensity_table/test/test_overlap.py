import numpy as np

from starfish import IntensityTable
from starfish.core.types import Coordinates, Features
from starfish.core.types._constants import OverlapStrategy
from .factories import create_intensity_table_with_coords
from ..overlap import (
    Area,
    find_overlaps_of_xarrays,
    remove_area_of_xarray,
    sel_area_of_xarray
)


def test_find_area_intersection():
    """
    Create various Area objects and verify their intersection are calculated correctly
    """
    area1 = Area(min_x=0, max_x=2, min_y=0, max_y=2)
    area2 = Area(min_x=1, max_x=2, min_y=1, max_y=3)
    intersection = Area.find_intersection(area1, area2)
    # intersection should be area with bottom point (1,1) and top point (2,2)
    assert intersection == Area(min_x=1, max_x=2, min_y=1, max_y=2)

    area2 = Area(min_x=3, max_x=5, min_y=3, max_y=5)
    intersection = Area.find_intersection(area1, area2)
    # no intersection
    assert intersection is None

    area2 = Area(min_x=0, max_x=5, min_y=3, max_y=5)
    intersection = Area.find_intersection(area1, area2)
    # area 2 right above area one
    assert intersection is None

    # try negatives
    area1 = Area(min_x=-1, max_x=1, min_y=0, max_y=2)
    area2 = Area(min_x=0, max_x=2, min_y=0, max_y=2)
    intersection = Area.find_intersection(area1, area2)
    assert intersection == Area(min_x=0, max_x=1, min_y=0, max_y=2)

    area2 = Area(min_x=-3, max_x=-2, min_y=0, max_y=2)
    intersection = Area.find_intersection(area1, area2)
    assert intersection is None


def test_find_overlaps_of_xarrays():
    """
    Create a list of overlapping IntensityTables and verify we identify the correct
    overlapping sections
    """
    # Create some overlapping intensity tables
    it0 = create_intensity_table_with_coords(Area(min_x=0, max_x=1,
                                                  min_y=0, max_y=1))
    it1 = create_intensity_table_with_coords(Area(min_x=.5, max_x=2,
                                                  min_y=.5, max_y=1.5))
    it2 = create_intensity_table_with_coords(Area(min_x=1.5, max_x=2.5,
                                                  min_y=0, max_y=1))
    it3 = create_intensity_table_with_coords(Area(min_x=0, max_x=1,
                                                  min_y=1, max_y=2))
    overlaps = find_overlaps_of_xarrays([it0, it1, it2, it3])
    # should have 4 total overlaps
    assert len(overlaps) == 4
    # overlap 1 between it0 and it1:
    assert (0, 1) in overlaps
    # overlap 1 between it0 and it1:
    assert (1, 2) in overlaps
    # overlap 3 between it1 and it3
    assert (1, 3) in overlaps
    # overlap 4 between it0 and it3
    assert (0, 3) in overlaps


def test_remove_area_of_xarray():
    """
    Tests removing a section of an IntensityTable defined by its physical area
    """
    it = create_intensity_table_with_coords(Area(min_x=0, max_x=2,
                                                 min_y=0, max_y=2), n_spots=10)

    area = Area(min_x=1, max_x=2, min_y=1, max_y=3)
    # grab some random coord values in this range
    removed_x = it.where(it.xc > 1, drop=True)[Coordinates.X.value].data[0]
    removed_y = it.where(it.yc > 1, drop=True)[Coordinates.X.value].data[3]

    it = remove_area_of_xarray(it, area)
    # assert coords from removed section are no longer in it
    assert not np.any(np.isclose(it[Coordinates.X.value], removed_x))
    assert not np.any(np.isclose(it[Coordinates.Y.value], removed_y))


def test_sel_area_of_xarray():
    """
    Tests selecting a section of an IntensityTable defined by its physical area
    """
    it = create_intensity_table_with_coords(Area(min_x=0, max_x=2, min_y=0, max_y=2), n_spots=10)

    area = Area(min_x=1, max_x=2, min_y=1, max_y=3)
    it = sel_area_of_xarray(it, area)

    # Assert new min/max values
    assert min(it[Coordinates.X.value]).data >= 1
    assert max(it[Coordinates.X.value]).data <= 2
    assert min(it[Coordinates.Y.value]).data >= 1
    assert max(it[Coordinates.X.value]).data <= 2


def test_take_max():
    """
    Create two overlapping IntensityTables with differing number of spots and verify that
    by concatenating them with the TAKE_MAX strategy we only include spots in the overlapping
    section from the IntensityTable that had the most.
    """
    it1 = create_intensity_table_with_coords(Area(min_x=0, max_x=2,
                                                  min_y=0, max_y=2), n_spots=10)
    it2 = create_intensity_table_with_coords(Area(min_x=1, max_x=2,
                                                  min_y=1, max_y=3), n_spots=20)

    concatenated = IntensityTable.concatenate_intensity_tables(
        [it1, it2], overlap_strategy=OverlapStrategy.TAKE_MAX)

    # The overlap section hits half of the spots from each intensity table, 5 from it1
    # and 10 from i21. It2 wins and the resulting concatenated table should have all the
    # spots from it2 (20) and 6 (one on the border) from it1 (6) for a total of 26 spots
    assert concatenated.sizes[Features.AXIS] == 26
