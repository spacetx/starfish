import warnings

import numpy as np
import pytest
from skimage import img_as_float32

from starfish.errors import DataFormatWarning
from starfish.imagestack.imagestack import ImageStack

NUM_ROUND = 2
NUM_CH = 2
NUM_Z = 2
HEIGHT = 10
WIDTH = 10


def create_tile_data_provider(dtype: np.number, corner_dtype: np.number):
    """
    Makes a stack that's all of the same type, except the round=0,ch=0,z=0 corner, which is a
    different type.  All the tiles are initialized with ones.

    Parameters
    ----------
    dtype : np.number
        The data type of all the tiles except the hyd=0,ch=0,z=0 corner.
    corner_dtype
        The data type of the tile in the hyd=0,ch=0,z=0 corner.

    Returns
    -------
    ImageStack :
        The image stack with the tiles initialized as described.
    """
    def tile_data_provider(round_: int, ch: int, z: int, height: int, width: int) -> np.ndarray:
        if round_ == 0 and ch == 0 and z == 0:
            return np.ones((height, width), dtype=corner_dtype)
        else:
            return np.ones((height, width), dtype=dtype)
    return tile_data_provider


def test_multiple_tiles_of_different_kind():
    with pytest.raises(TypeError):
        ImageStack.synthetic_stack(
            NUM_ROUND, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(np.uint32, np.float32),
        )


def test_multiple_tiles_of_same_dtype():
    stack = ImageStack.synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_data_provider=create_tile_data_provider(np.uint32, np.uint32),
    )
    expected = np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.uint32)
    assert np.array_equal(stack.numpy_array, img_as_float32(expected))


def test_int_type_promotion():
    with warnings.catch_warnings(record=True) as warnings_:
        stack = ImageStack.synthetic_stack(
            NUM_ROUND, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(np.int32, np.int8),
        )
        assert len(warnings_) == 2
        assert issubclass(warnings_[0].category, DataFormatWarning)
        assert issubclass(warnings_[1].category, UserWarning)
    expected = img_as_float32(np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.int32))
    corner = img_as_float32(np.ones(
        (HEIGHT,
         WIDTH), dtype=np.int8))
    expected[0, 0, 0] = corner
    assert np.array_equal(stack.numpy_array, img_as_float32(expected))


def test_uint_type_promotion():
    with warnings.catch_warnings(record=True) as warnings_:
        stack = ImageStack.synthetic_stack(
            NUM_ROUND, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(np.uint32, np.uint8),
        )
        assert len(warnings_) == 2
        assert issubclass(warnings_[0].category, DataFormatWarning)
        assert issubclass(warnings_[1].category, UserWarning)
    expected = img_as_float32(np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.uint32))
    corner = img_as_float32(np.ones(
        (HEIGHT,
         WIDTH), dtype=np.uint8))
    expected[0, 0, 0] = corner
    assert np.array_equal(stack.numpy_array, img_as_float32(expected))


def test_float_type_demotion():
    with warnings.catch_warnings(record=True) as warnings_:
        stack = ImageStack.synthetic_stack(
            NUM_ROUND, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_data_provider=create_tile_data_provider(np.float64, np.float32),
        )
        assert len(warnings_) == 2
        assert issubclass(warnings_[0].category, DataFormatWarning)
        assert issubclass(warnings_[1].category, UserWarning)
    expected = np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.float64)
    assert np.array_equal(stack.numpy_array, expected)
