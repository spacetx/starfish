from collections import OrderedDict

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Coordinates, Indices


def test_indexing_by_r_ch_z():
    stack = ImageStack.synthetic_stack(num_round=5, num_ch=5, num_z=15,
                                       tile_height=200, tile_width=200)

    indexed = stack.sel({Indices.ROUND: (1, None), Indices.CH: 0, Indices.Z: 0})
    # assert that the shape changed
    expected_shape = OrderedDict([(Indices.ROUND, 4), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    indexed = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 0})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    indexed = stack.sel({Indices.ROUND: 1, Indices.CH: (3, None)})
    expected_shape = OrderedDict(
        [(Indices.ROUND, 1), (Indices.CH, 2), (Indices.Z, 15), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape

    indexed = stack.sel({Indices.ROUND: 1, Indices.CH: (None, 3), Indices.Z: (7, None)})
    expected_shape = OrderedDict(
        [(Indices.ROUND, 1), (Indices.CH, 3), (Indices.Z, 8), (Indices.Y, 200), (Indices.X, 200)])
    assert indexed.shape == expected_shape


def set_coordinate_values(stack, coordinates):
    """Sets the same coordinate values on each tile """
    for _round in range(stack.num_rounds):
        for ch in range(stack.num_chs):
            for z in range(stack.num_zlayers):
                coordinate_selector = {
                    Indices.ROUND.value: _round,
                    Indices.CH.value: ch,
                    Indices.Z.value: z,
                }
                coordinates_values = [coordinates[Coordinates.X][0], coordinates[Coordinates.X][1],
                                      coordinates[Coordinates.Y][0], coordinates[Coordinates.Y][1],
                                      coordinates[Coordinates.Z][0], coordinates[Coordinates.Z][1]]
                stack._coordinates.loc[coordinate_selector] = np.array(coordinates_values)


def test_indexing_by_x_y():
    stack = ImageStack.synthetic_stack(num_round=1, num_ch=2, num_z=2,
                                       tile_height=200, tile_width=200)

    set_coordinate_values(stack, {Coordinates.X: (1, 2),
                                  Coordinates.Y: (4, 6),
                                  Coordinates.Z: (1, 3)})

    # index on single value of y
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: 100, Indices.X: (None, 100)})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 1), (Indices.X, 100)])
    assert indexed_stack.shape == expected_shape

    # y should be range of one pixel size where pixel size = (ymax-ymin) / size of y dimension
    # (6-4) / 200 = .01
    expected_coords = {Coordinates.X: (1, 1.505), Coordinates.Y: (5, 5.01), Coordinates.Z: (1, 3)}

    check_coodinate_values(indexed_stack, expected_coords)

    # indexed on first half of x and y dimensions:
    indexed_stack = stack.sel({Indices.Y: (None, 100), Indices.X: (None, 100)})

    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 2),
                                  (Indices.Z, 2), (Indices.Y, 100), (Indices.X, 100)])
    assert indexed_stack.shape == expected_shape

    # xmin = 1, xmax = 1.505 (+ .005 size of one x physical pixel)
    # ymin = 4, ymax=5.01 (+ .01 size of one y physical pixel)
    expected_coords = {Coordinates.X: (1, 1.505), Coordinates.Y: (4, 5.01), Coordinates.Z: (1, 3)}

    check_coodinate_values(indexed_stack, expected_coords)

    # index on single x and y
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: 100, Indices.X: 150})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 1), (Indices.X, 1)])
    assert indexed_stack.shape == expected_shape

    # y should be range of one pixel size where pixel size = (ymax-ymin) / size of y dimension
    # (2-1) / 200 = .005
    expected_coords = {Coordinates.X: (1.75, 1.755),
                       Coordinates.Y: (5, 5.01),
                       Coordinates.Z: (1, 3)}
    check_coodinate_values(indexed_stack, expected_coords)

    # Negative indexing
    indexed_stack = stack.sel({Indices.ROUND: 0, Indices.CH: 0, Indices.Z: 1,
                               Indices.Y: (None, -10), Indices.X: (None, -10)})
    expected_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                  (Indices.Z, 1), (Indices.Y, 190), (Indices.X, 190)])
    assert indexed_stack.shape == expected_shape
    expected_coords = {Coordinates.X: (1, 1.955),
                       Coordinates.Y: (4, 5.91),
                       Coordinates.Z: (1, 3)}
    check_coodinate_values(indexed_stack, expected_coords)


def check_coodinate_values(stack, expected_coords):
    for _round in range(stack.num_rounds):
        for ch in range(stack.num_chs):
            for z in range(stack.num_zlayers):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }

                xmin, xmax = stack.coordinates(indices, Coordinates.X)
                ymin, ymax = stack.coordinates(indices, Coordinates.Y)
                zmin, zmax = stack.coordinates(indices, Coordinates.Z)

                expected_xmin, expected_xmax = expected_coords[Coordinates.X]
                expected_ymin, expected_ymax = expected_coords[Coordinates.Y]
                expected_zmin, expected_zmax = expected_coords[Coordinates.Z]

                assert np.isclose(xmin, expected_xmin)
                assert np.isclose(xmax, expected_xmax)
                assert np.isclose(ymin, expected_ymin)
                assert np.isclose(ymax, expected_ymax)
                assert np.isclose(zmin, expected_zmin)
                assert np.isclose(zmax, expected_zmax)
