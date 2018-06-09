import warnings

import numpy
import pytest
from slicedimage import Tile, TileSet

from starfish.constants import Coordinates, Indices
from starfish.errors import DataFormatWarning
from starfish.image import ImageStack


NUM_HYB = 2
NUM_CH = 2
NUM_Z = 2
HEIGHT = 10
WIDTH = 10


def make_stack(dtype: numpy.number, corner_dtype: numpy.number):
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
            Indices.HYB: NUM_HYB,
            Indices.CH: NUM_CH,
            Indices.Z: NUM_Z,
        },
        default_tile_shape=(HEIGHT, WIDTH),
    )
    for hyb in range(NUM_HYB):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
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
                    data = numpy.ones((HEIGHT, WIDTH), dtype=corner_dtype)
                else:
                    data = numpy.ones((HEIGHT, WIDTH), dtype=dtype)
                tile.numpy_array = data

                img.add_tile(tile)

    return ImageStack(img)


def test_different_kind():
    with pytest.raises(TypeError):
        make_stack(numpy.uint32, numpy.float32)


def test_all_identical():
    stack = make_stack(numpy.uint32, numpy.uint32)
    assert stack.numpy_array.all() == 1


def test_int_type_promotion():
    with warnings.catch_warnings(record=True) as w:
        stack = make_stack(numpy.int32, numpy.int8)
        assert len(w) == 1
        assert issubclass(w[0].category, DataFormatWarning)
    expected = numpy.empty(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.int32)
    expected.fill(16777216)
    corner = numpy.ones(
        (HEIGHT,
         WIDTH), dtype=numpy.int32)
    expected[0, 0, 0] = corner
    assert stack.numpy_array.all() == expected.all()


def test_float_type_promotion():
    with warnings.catch_warnings(record=True) as w:
        stack = make_stack(numpy.float64, numpy.float32)
        assert len(w) == 1
        assert issubclass(w[0].category, DataFormatWarning)
    expected = numpy.empty(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.int64)
    expected.fill(2.0)
    corner = numpy.ones(
        (HEIGHT,
         WIDTH), dtype=numpy.int64)
    expected[0, 0, 0] = corner
    assert stack.numpy_array.all() == expected.all()
