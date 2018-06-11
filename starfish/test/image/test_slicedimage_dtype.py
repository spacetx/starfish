import warnings

import numpy
import pytest

from starfish.errors import DataFormatWarning
from starfish.test.dataset_fixtures import synthetic_stack


NUM_HYB = 2
NUM_CH = 2
NUM_Z = 2
HEIGHT = 10
WIDTH = 10


def create_tile_data_provider(dtype: numpy.number, corner_dtype: numpy.number):
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
    def tile_data_provider(hyb: int, ch: int, z: int, height: int, width: int) -> numpy.ndarray:
        if hyb == 0 and ch == 0 and z == 0:
            return numpy.ones((height, width), dtype=corner_dtype)
        else:
            return numpy.ones((height, width), dtype=dtype)
    return tile_data_provider


def test_multiple_tiles_of_different_kind():
    with pytest.raises(TypeError):
        synthetic_stack(
            NUM_HYB, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(numpy.uint32, numpy.float32),
        )


def test_multiple_tiles_of_same_dtype():
    stack = synthetic_stack(
        NUM_HYB, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_data_provider=create_tile_data_provider(numpy.uint32, numpy.uint32),
    )
    expected = numpy.ones(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.uint32)
    assert numpy.array_equal(stack.numpy_array, expected)


def test_int_type_promotion():
    with warnings.catch_warnings(record=True) as w:
        stack = synthetic_stack(
            NUM_HYB, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(numpy.int32, numpy.int8),
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DataFormatWarning)
    expected = numpy.ones(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.int32)
    corner = numpy.empty(
        (HEIGHT,
         WIDTH), dtype=numpy.int32)
    corner.fill(16777216)
    expected[0, 0, 0] = corner
    assert numpy.array_equal(stack.numpy_array, expected)


def test_uint_type_promotion():
    with warnings.catch_warnings(record=True) as w:
        stack = synthetic_stack(
            NUM_HYB, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(numpy.uint32, numpy.uint8),
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DataFormatWarning)
    expected = numpy.ones(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.uint32)
    corner = numpy.empty(
        (HEIGHT,
         WIDTH), dtype=numpy.uint32)
    corner.fill(16777216)
    expected[0, 0, 0] = corner
    assert numpy.array_equal(stack.numpy_array, expected)


def test_float_type_promotion():
    with warnings.catch_warnings(record=True) as w:
        stack = synthetic_stack(
            NUM_HYB, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(numpy.float64, numpy.float32),
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DataFormatWarning)
    expected = numpy.ones(
        (NUM_HYB,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=numpy.float64)
    assert numpy.array_equal(stack.numpy_array, expected)
