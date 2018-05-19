import unittest

import numpy
from slicedimage import Tile, TileSet

from starfish.constants import Coordinates, Indices
from starfish.image import ImageStack


class TestSetSliceAPI(unittest.TestCase):
    NUM_HYB = 2
    NUM_CH = 3
    NUM_Z = 4
    HEIGHT = 30
    WIDTH = 20

    def setUp(self):
        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
            {
                Indices.HYB: TestSetSliceAPI.NUM_HYB,
                Indices.CH: TestSetSliceAPI.NUM_CH,
                Indices.Z: TestSetSliceAPI.NUM_Z,
            },
            default_tile_shape=(TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH),
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
                    tile.numpy_array = numpy.zeros((TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH))

                    img.add_tile(tile)

        self.stack = ImageStack(img)

    def test_set_slice_simple_index(self):
        """
        Sets a slice across one of the indices at the end.  For instance, if the dimensions are
        (P, Q0,..., Qn-1, R), sets a slice across either P or R.
        """
        hyb = 1
        self.stack.set_slice(
            {Indices.HYB: hyb},
            numpy.ones((TestSetSliceAPI.NUM_CH, TestSetSliceAPI.NUM_Z, TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH)))

        expected = numpy.zeros(
            (TestSetSliceAPI.NUM_HYB, TestSetSliceAPI.NUM_CH, TestSetSliceAPI.NUM_Z,
             TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH))
        expected[hyb].fill(1)
        self.assertEqual(self.stack.numpy_array.all(), expected.all())

    def test_set_slice_middle_index(self):
        """
        Sets a slice across one of the indices in the middle.  For instance, if the dimensions are
        (P, Q0,..., Qn-1, R), slice across one of the Q axes.
        """
        ch = 1
        self.stack.set_slice(
            {Indices.CH: ch},
            numpy.ones((TestSetSliceAPI.NUM_HYB, TestSetSliceAPI.NUM_Z, TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH)))

        expected = numpy.zeros(
            (TestSetSliceAPI.NUM_HYB, TestSetSliceAPI.NUM_CH, TestSetSliceAPI.NUM_Z,
             TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH))
        expected[:, ch].fill(1)
        self.assertEqual(self.stack.numpy_array.all(), expected.all())

    def test_set_slice_range(self):
        """
        Sets a slice across a range of one of the dimensions.
        """
        zrange = slice(1, 3)
        self.stack.set_slice(
            {Indices.Z: zrange},
            numpy.ones(
                (TestSetSliceAPI.NUM_HYB, TestSetSliceAPI.NUM_CH, zrange.stop - zrange.start,
                 TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH)))

        expected = numpy.zeros(
            (TestSetSliceAPI.NUM_HYB, TestSetSliceAPI.NUM_CH, zrange.stop - zrange.start,
             TestSetSliceAPI.HEIGHT, TestSetSliceAPI.WIDTH))
        expected[:, :, zrange.start:zrange.stop].fill(1)
        self.assertEqual(self.stack.numpy_array.all(), expected.all())
