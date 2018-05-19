import unittest

import numpy
from slicedimage import Tile, TileSet

from starfish.constants import Coordinates, Indices
from starfish.image import ImageStack


def multiply(array, value):
    return array * value


class TestSetSliceAPI(unittest.TestCase):
    NUM_HYB = 2
    NUM_CH = 3
    NUM_Z = 4
    Y = 30
    X = 20

    def setUp(self):
        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
            {
                Indices.HYB: TestSetSliceAPI.NUM_HYB,
                Indices.CH: TestSetSliceAPI.NUM_CH,
                Indices.Z: TestSetSliceAPI.NUM_Z,
            },
            default_tile_shape=(TestSetSliceAPI.Y, TestSetSliceAPI.X),
        )
        for hyb in range(TestSetSliceAPI.NUM_HYB):
            for ch in range(TestSetSliceAPI.NUM_CH):
                for z in range(TestSetSliceAPI.NUM_Z):
                    tile = Tile(
                        {
                            Coordinates.X: (0.0, 0.001),
                            Coordinates.Y: (0.0, 0.001),
                            Coordinates.Z: (0.0, 0.001),
                        },
                        {
                            Indices.HYB: hyb,
                            Indices.CH: ch,
                            Indices.Z: z,
                        }
                    )
                    tile.numpy_array = numpy.ones(
                        (TestSetSliceAPI.Y, TestSetSliceAPI.X))

                    img.add_tile(tile)

        self.stack = ImageStack(img)

    def test_apply(self):
        self.stack.apply(multiply, value=2)
        assert (self.stack.numpy_array == 2).all()

    def test_apply_3d(self):
        self.stack.apply(multiply, value=4, is_volume=True)
        assert (self.stack.numpy_array == 4).all()
