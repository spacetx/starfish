from collections import OrderedDict

from starfish.imagestack import indexing_utils
from starfish.imagestack.imagestack import ImageStack
from starfish.test import test_utils
from starfish.types import Axes, Coordinates, PhysicalCoordinateTypes


def test_imagestack_indexing():
    """Tests indexing on an Imagestack with a shape (5, 5, 15, 200, 200)
        steps:
        1.) stack.sel(indexers)
        2.) assert new shape of stack is what we expect

    """

    stack = ImageStack.synthetic_stack(num_round=5, num_ch=5, num_z=15,
                                       tile_height=200, tile_width=200)

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


def test_sel_by_physical_coords():
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 10), (Axes.X, 10)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = test_utils.imagestack_with_coords_factory(stack_shape, physical_coords)
    # x_coords = [1. 1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
    #  1.66666667 1.77777778 1.88888889 2.]
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.X.value], 1.2) == 2
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.X.value], 1.5) == 4
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.X.value], (1.2, 1.5)) == (2, 4)
    new_x_size = 4 - 2

    # y_coords = [4. 4.22222222 4.44444444 4.66666667 4.88888889 5.11111111
    #  5.33333333 5.55555556 5.77777778 6. ]
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.Y.value], 4) == 0
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.Y.value], 5.1) == 5
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.Y.value], (4, 5.1)) == (0, 5)
    new_y_size = 5 - 0

    indexed_stack = stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 0,
                               Coordinates.X: (1.2, 1.5), Coordinates.Y: (4, 5.1)})
    
    expected_shape = OrderedDict([(Axes.ROUND, 1), (Axes.CH, 1),
                                  (Axes.ZPLANE, 1), (Axes.Y, new_y_size), (Axes.X, new_x_size)])
    assert indexed_stack.shape == expected_shape

    # assert values outside the range are given min/max of array
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.X.value], 5) == 9
    assert indexing_utils.find_nearest(stack.xarray[Coordinates.X.value], -5) == 0
