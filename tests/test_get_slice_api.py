import unittest

import numpy
from slicedimage import Tile, TileSet

from starfish.constants import Coordinates, Indices
from starfish.image import ImageStack


class TestGetSliceAPI(unittest.TestCase):
    NUM_HYB = 2
    NUM_CH = 3
    NUM_Z = 4
    HEIGHT = 30
    WIDTH = 20

    def setUp(self):
        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
            {
                Indices.HYB: TestGetSliceAPI.NUM_HYB,
                Indices.CH: TestGetSliceAPI.NUM_CH,
                Indices.Z: TestGetSliceAPI.NUM_Z,
            },
            default_tile_shape=(TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH),
        )
        for hyb in range(TestGetSliceAPI.NUM_HYB):
            for ch in range(TestGetSliceAPI.NUM_CH):
                for z in range(TestGetSliceAPI.NUM_Z):
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
                    data = numpy.empty((TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH))
                    data.fill((hyb * TestGetSliceAPI.NUM_CH + ch) * TestGetSliceAPI.NUM_Z + z)
                    tile.numpy_array = data

                    img.add_tile(tile)

        self.stack = ImageStack(img)

    def test_get_slice_simple_index(self):
        """
        Retrieve a slice across one of the indices at the end.  For instance, if the dimensions are
        (P, Q0,..., Qn-1, R), slice across either P or R.
        """
        hyb = 1
        imageslice, axes = self.stack.get_slice(
            {Indices.HYB: hyb}
        )
        self.assertEqual(
            imageslice.shape,
            (TestGetSliceAPI.NUM_CH, TestGetSliceAPI.NUM_Z, TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH,))
        self.assertEqual(axes, [Indices.CH, Indices.Z])

        for ch in range(TestGetSliceAPI.NUM_CH):
            for z in range(TestGetSliceAPI.NUM_Z):
                data = numpy.empty((TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH))
                data.fill((hyb * TestGetSliceAPI.NUM_CH + ch) * TestGetSliceAPI.NUM_Z + z)

                self.assertEqual(data.all(), imageslice[ch, z].all())

    def test_get_slice_middle_index(self):
        """
        Retrieve a slice across one of the indices in the middle.  For instance, if the dimensions are
        (P, Q0,..., Qn-1, R), slice across one of the Q axes.
        """
        ch = 1
        imageslice, axes = self.stack.get_slice(
            {Indices.CH: ch}
        )
        self.assertEqual(
            imageslice.shape,
            (TestGetSliceAPI.NUM_HYB, TestGetSliceAPI.NUM_Z, TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH,))
        self.assertEqual(axes, [Indices.HYB, Indices.Z])

        for hyb in range(TestGetSliceAPI.NUM_HYB):
            for z in range(TestGetSliceAPI.NUM_Z):
                data = numpy.empty((TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH))
                data.fill((hyb * TestGetSliceAPI.NUM_CH + ch) * TestGetSliceAPI.NUM_Z + z)

                self.assertEqual(data.all(), imageslice[hyb, z].all())

    def test_get_slice_range(self):
        """
        Retrieve a slice across a range of one of the dimensions.
        """
        zrange = slice(1, 3)
        imageslice, axes = self.stack.get_slice(
            {Indices.Z: zrange}
        )
        self.assertEqual(
            imageslice.shape,
            (TestGetSliceAPI.NUM_HYB, TestGetSliceAPI.NUM_CH, 2, TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH,))
        self.assertEqual(axes, [Indices.HYB, Indices.CH, Indices.Z])

        for hyb in range(TestGetSliceAPI.NUM_HYB):
            for ch in range(TestGetSliceAPI.NUM_CH):
                for z in range(zrange.stop - zrange.start):
                    data = numpy.empty((TestGetSliceAPI.HEIGHT, TestGetSliceAPI.WIDTH))
                    data.fill((hyb * TestGetSliceAPI.NUM_CH + ch) * TestGetSliceAPI.NUM_Z + (z + zrange.start))

                    self.assertEqual(data.all(), imageslice[hyb, ch, z].all())
