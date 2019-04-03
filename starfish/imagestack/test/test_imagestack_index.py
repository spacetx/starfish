from collections import OrderedDict

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes


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
