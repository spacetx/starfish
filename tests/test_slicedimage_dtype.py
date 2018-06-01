import unittest
import warnings

import numpy
from slicedimage import Tile, TileSet

from starfish.constants import Coordinates, Indices
from starfish.errors import DataFormatWarning
from starfish.image import ImageStack


class TestSlicedImageDtype(unittest.TestCase):
    NUM_HYB = 2
    NUM_CH = 2
    NUM_Z = 2
    HEIGHT = 10
    WIDTH = 10

    def make_stack(self, dtype: numpy.number, corner_dtype: numpy.number):
        """
        Makes a stack that's all of the same type, except the hyb=0,ch=0,z=0 corner, which is a different type.  All the
        tiles are initialized with ones.

        Parameters
        ----------
        dtype : numpy.number
            The data type of all the tiles except the hyd=0,ch=0,z=0 corner.
        corner_dtype
            The data type of the tile in the hyd=0,ch=0,z=0 corner.

        Returns
        -------
        ImageStack :
            The image stack with the tiles initialized as described.
        """
        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
            {
                Indices.HYB: TestSlicedImageDtype.NUM_HYB,
                Indices.CH: TestSlicedImageDtype.NUM_CH,
                Indices.Z: TestSlicedImageDtype.NUM_Z,
            },
            default_tile_shape=(TestSlicedImageDtype.HEIGHT, TestSlicedImageDtype.WIDTH),
        )
        for hyb in range(TestSlicedImageDtype.NUM_HYB):
            for ch in range(TestSlicedImageDtype.NUM_CH):
                for z in range(TestSlicedImageDtype.NUM_Z):
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
                    if hyb == 0 and ch == 0 and z == 0:
                        data = numpy.ones((TestSlicedImageDtype.HEIGHT, TestSlicedImageDtype.WIDTH), dtype=corner_dtype)
                    else:
                        data = numpy.ones((TestSlicedImageDtype.HEIGHT, TestSlicedImageDtype.WIDTH), dtype=dtype)
                    tile.numpy_array = data

                    img.add_tile(tile)

        return ImageStack(img)

    def test_different_kind(self):
        with self.assertRaises(TypeError):
            self.make_stack(numpy.uint32, numpy.float32)

    def test_all_identical(self):
        stack = self.make_stack(numpy.uint32, numpy.uint32)
        self.assertEqual(stack.numpy_array.all(), 1)

    def test_int_type_promotion(self):
        with warnings.catch_warnings(record=True) as w:
            stack = self.make_stack(numpy.int32, numpy.int8)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DataFormatWarning))
        expected = numpy.empty(
            (TestSlicedImageDtype.NUM_HYB,
             TestSlicedImageDtype.NUM_CH,
             TestSlicedImageDtype.NUM_Z,
             TestSlicedImageDtype.HEIGHT,
             TestSlicedImageDtype.WIDTH), dtype=numpy.int32)
        expected.fill(16777216)
        corner = numpy.ones(
            (TestSlicedImageDtype.HEIGHT,
             TestSlicedImageDtype.WIDTH), dtype=numpy.int32)
        expected[0, 0, 0] = corner
        self.assertEqual(stack.numpy_array.all(), expected.all())

    def test_float_type_promotion(self):
        with warnings.catch_warnings(record=True) as w:
            stack = self.make_stack(numpy.float64, numpy.float32)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DataFormatWarning))
        expected = numpy.empty(
            (TestSlicedImageDtype.NUM_HYB,
             TestSlicedImageDtype.NUM_CH,
             TestSlicedImageDtype.NUM_Z,
             TestSlicedImageDtype.HEIGHT,
             TestSlicedImageDtype.WIDTH), dtype=numpy.int64)
        expected.fill(2.0)
        corner = numpy.ones(
            (TestSlicedImageDtype.HEIGHT,
             TestSlicedImageDtype.WIDTH), dtype=numpy.int64)
        expected[0, 0, 0] = corner
        self.assertEqual(stack.numpy_array.all(), expected.all())
