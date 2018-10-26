from collections import OrderedDict

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Indices


def test_imagestack_indexing():
    """Tests indexing on an Imagestack with a shape (5, 5, 15, 200, 200)
        steps:
        1.) stack.sel(indexers)
        2.) assert new shape of stack is what we expect

    """

    stack = ImageStack.synthetic_stack(num_round=5, num_ch=5, num_z=15,
                                       tile_height=200, tile_width=200)

    # index on range of rounds and single ch and Z
    indexed = stack.sel({Indices.ROUND: (1, None), Indices.CH: 0, Indices.Z: 0})
    expected_shape = OrderedDict([(Indices.ROUND, 4), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round ch and z
    indexed = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 0})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round and range of ch
    indexed = stack.sel({Indices.ROUND: 1, Indices.CH: (3, None)})
    expected_shape = OrderedDict(
        [(Indices.ROUND, 1), (Indices.CH, 2), (Indices.Z, 15), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    # index on single round and range of ch and Z
    indexed = stack.sel({Indices.ROUND: 1, Indices.CH: (None, 3), Indices.Z: (7, None)})
    expected_shape = OrderedDict(
        [(Indices.ROUND, 1), (Indices.CH, 3), (Indices.Z, 8), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    # index on first half of X and single value of Y
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: 100, Indices.X: (None, 100)})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 1), (Indices.X, 100)])
    assert indexed_stack.shape == expected_shape

    # index on first half of X and Y
    indexed_stack = stack.sel({Indices.Y: (None, 100), Indices.X: (None, 100)})

    expected_shape = OrderedDict([(Indices.ROUND, 5), (Indices.CH, 5),
                                  (Indices.Z, 15), (Indices.Y, 100), (Indices.X, 100)])
    assert indexed_stack.shape == expected_shape

    # index on single x and y
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: 100, Indices.X: 150})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 1), (Indices.X, 1)])
    assert indexed_stack.shape == expected_shape

    # Negative indexing
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: (None, -10), Indices.X: (None, -10)})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 190), (Indices.X, 190)])
    assert indexed_stack.shape == expected_shape
