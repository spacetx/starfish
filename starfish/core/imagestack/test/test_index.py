import os
import tempfile
from collections import OrderedDict

import numpy as np

from starfish.core.imagestack import indexing_utils as iu
from starfish.core.types import Axes, Coordinates, PhysicalCoordinateTypes
from .factories import imagestack_with_coords_factory, synthetic_stack
from ..imagestack import ImageStack


def test_imagestack_indexing():
    """Tests indexing on an Imagestack with a shape (5, 5, 15, 200, 200)
        steps:
        1.) stack.sel(indexers)
        2.) assert new shape of stack is what we expect

    """

    stack = synthetic_stack(num_round=5, num_ch=5, num_z=15, tile_height=200, tile_width=200)

    # index on range of rounds and single ch and Z
    indexed = stack.sel({Axes.ROUND: (1, None), Axes.CH: 0, Axes.ZPLANE: 0})
    expected_shape = OrderedDict([(Axes.ROUND, 4), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 200), (Axes.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round ch and z
    indexed = stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 0})
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 200), (Axes.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round and range of ch
    indexed = stack.sel({Axes.ROUND: 1, Axes.CH: (3, None)})
    expected_shape = OrderedDict(
        [(Axes.ROUND, 1), (Axes.CH, 2), (Axes.ZPLANE, 15), (Axes.Y, 200), (Axes.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round and range of ch and Z
    indexed = stack.sel({Axes.ROUND: 1, Axes.CH: (None, 3), Axes.ZPLANE: (7, None)})
    expected_shape = OrderedDict(
        [(Axes.ROUND, 1), (Axes.CH, 4), (Axes.ZPLANE, 8), (Axes.Y, 200), (Axes.X, 200)])
    assert indexed.shape == expected_shape

    # index on first half of X and single value of Y
    indexed_stack = stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 1,
                               Axes.Y: 100, Axes.X: (None, 100)})
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 1), (Axes.X, 100)])
    assert indexed_stack.shape == expected_shape

    # index on first half of X and Y
    indexed_stack = stack.sel({Axes.Y: (None, 100), Axes.X: (None, 100)})

    expected_shape = OrderedDict([(Axes.ROUND, 5), (Axes.CH, 5),
                                  (Axes.ZPLANE, 15), (Axes.Y, 100), (Axes.X, 100)])
    assert indexed_stack.shape == expected_shape

    # index on single x and y
    indexed_stack = stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 1,
                               Axes.Y: 100, Axes.X: 150})
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 1), (Axes.X, 1)])
    assert indexed_stack.shape == expected_shape

    # Negative indexing
    indexed_stack = stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 1,
                               Axes.Y: (None, -10), Axes.X: (None, -10)})
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 190), (Axes.X, 190)])
    assert indexed_stack.shape == expected_shape


X_COORDS = 1, 2
Y_COORDS = 4, 6
Z_COORDS = 1, 3


def test_find_nearest():
    """
    Set up ImageStack with physical coordinates:
        x_coords = [1. 1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
        1.66666667 1.77777778 1.88888889 2.]

        y_coords = [4. 4.22222222 4.44444444 4.66666667 4.88888889 5.11111111
        5.33333333 5.55555556 5.77777778 6. ]

     Test that find_nearest() finds the correct corresponding positional index values
    """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 10), (Axes.X, 10)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)
    assert iu.find_nearest(stack.xarray[Coordinates.X.value], 1.2) == 2
    assert iu.find_nearest(stack.xarray[Coordinates.X.value], 1.5) == 4
    assert iu.find_nearest(stack.xarray[Coordinates.X.value], (1.2, 1.5)) == (2, 4)

    assert iu.find_nearest(stack.xarray[Coordinates.Y.value], 4) == 0
    assert iu.find_nearest(stack.xarray[Coordinates.Y.value], 5.1) == 5
    assert iu.find_nearest(stack.xarray[Coordinates.Y.value], (4, 5.1)) == (0, 5)

    # assert values outside the range are given min/max of array
    assert iu.find_nearest(stack.xarray[Coordinates.X.value], 5) == 9
    assert iu.find_nearest(stack.xarray[Coordinates.X.value], -5) == 0


def test_convert_coords_to_indices():
    """
    Set up ImageStack with physical coordinates:
        x_coords = [1. 1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
        1.66666667 1.77777778 1.88888889 2.]

        y_coords = [4. 4.22222222 4.44444444 4.66666667 4.88888889 5.11111111
        5.33333333 5.55555556 5.77777778 6. ]

    Test that convert_coords_to_indices() correctly converts Coordinate indices
    to their corresponding positional indices
    """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 10), (Axes.X, 10)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)
    coordinate_indices = {Coordinates.X: (1.2, 1.5), Coordinates.Y: (4, 5.1)}
    positional_indices = iu.convert_coords_to_indices(stack.xarray, coordinate_indices)

    assert positional_indices[Axes.X] == iu.find_nearest(
        stack.xarray[Coordinates.X.value], (1.2, 1.5))
    assert positional_indices[Axes.Y] == iu.find_nearest(
        stack.xarray[Coordinates.Y.value], (4, 5.1))


def test_sel_by_physical_coords():
    """
    Set up ImageStack with physical coordinates:
        x_coords = [1. 1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
        1.66666667 1.77777778 1.88888889 2.]

        y_coords = [4. 4.22222222 4.44444444 4.66666667 4.88888889 5.11111111
        5.33333333 5.55555556 5.77777778 6. ]

    Test that sel_by_physical_coords() correctly indexes the imagestack by the
    corresponding positional indexers
    """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 10), (Axes.X, 10)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)

    indexed_stack_by_coords = stack.sel_by_physical_coords({Coordinates.X: (1.2, 1.5),
                                                            Coordinates.Y: (4, 5.1)})
    indexed_stack_by_pos = stack.sel({Axes.X: (2, 4), Axes.Y: (0, 5)})

    # assert that the resulting xarrays are the same
    assert indexed_stack_by_coords.xarray.equals(indexed_stack_by_pos.xarray)


def test_sel_by_physical_and_axes():
    """
       Set up ImageStack with physical coordinates:
           x_coords = [1. 1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
           1.66666667 1.77777778 1.88888889 2.]

           y_coords = [4. 4.22222222 4.44444444 4.66666667 4.88888889 5.11111111
           5.33333333 5.55555556 5.77777778 6. ]

       Test that sel_by_physical_coords() correctly indexes the imagestack by the
       corresponding positional indexers
       """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 10), (Axes.X, 10)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)
    indexed_stack_by_coords = stack.sel_by_physical_coords({Coordinates.X: (1.2, 1.5),
                                                            Coordinates.Y: (4, 5.1)})
    indexed_stack = indexed_stack_by_coords.sel({Axes.ROUND: 2, Axes.CH: 1, Axes.ZPLANE: 0})
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, 5), (Axes.X, 2)])
    assert indexed_stack.shape == expected_shape


def test_nonindexed_dimensions_restored():
    """When the selection removes a dimension, xarray.expand_dims does not expand the non-indexed
    dimensions that were removed.  For example, if one selects only a single zplane, it reduce the z
    physical coordinate to a coordinate scalar, and not an array of size 1.  This verifies that the
    workaround we introduced to restore the dependent axes's labels is in place.
    """
    stack = synthetic_stack(num_round=5, num_ch=5, num_z=15, tile_height=200, tile_width=200)

    for selector in (
            {Axes.ROUND: 0, Axes.CH: 2, Axes.ZPLANE: 5},
            {Axes.ROUND: (0, 3), Axes.CH: 2, Axes.ZPLANE: 5},
            {Axes.CH: (None, 3), Axes.ZPLANE: 5},
    ):
        sel_xarray = stack.sel(selector).xarray

        # when the selection removes a dimension (e.g., only select a single z plane)
        for primary_axis, dependent_axis in (
                (Axes.X, Coordinates.X),
                (Axes.Y, Coordinates.Y),
                (Axes.ZPLANE, Coordinates.Z),
        ):
            assert len(sel_xarray[primary_axis.value]) == len(sel_xarray[dependent_axis.value])


def test_select_and_export():
    """Tests selecting on an Imagestack with a shape (5, 5, 15, 200, 200)
        1.) stack.sel(indexers)
        2.) export stack
    """
    stack = synthetic_stack(
        num_round=5, num_ch=5, num_z=15, tile_height=200, tile_width=200)

    # select on range of rounds and single ch and Z
    selected = stack.sel({Axes.ROUND: (1, None), Axes.CH: (2, 3), Axes.ZPLANE: 0})

    with tempfile.TemporaryDirectory() as tfd:
        path = os.path.join(tfd, "stack.json")
        selected.export(path)

        loaded = ImageStack.from_path_or_url(path)

        assert np.array_equal(selected.xarray, loaded.xarray)
        for coords in (
                Axes.ROUND, Axes.CH, Axes.ZPLANE, Coordinates.X, Coordinates.Y, Coordinates.Z):
            assert np.allclose(selected.xarray[coords.value], loaded.xarray[coords.value])
