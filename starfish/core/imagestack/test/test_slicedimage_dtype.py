import warnings
from typing import Mapping, Union

import numpy as np
import pytest
from skimage import img_as_float32
from slicedimage import ImageFormat

from starfish.core.errors import DataFormatWarning
from starfish.core.experiment.builder.providers import FetchedTile, TileFetcher
from starfish.core.types import Axes, Coordinates, CoordinateValue
from .factories import synthetic_stack

NUM_ROUND = 2
NUM_CH = 2
NUM_Z = 2
HEIGHT = 10
WIDTH = 10


class OnesTilesByDtype(FetchedTile):
    def __init__(self, dtype: np.number) -> None:
        super().__init__()
        self._dtype = dtype

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: HEIGHT, Axes.X: WIDTH}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=self._dtype)


class CornerDifferentDtype(TileFetcher):
    """
    TileFetcher that fetches tiles all of the same type, except the round=0,ch=0,z=0 corner, which
    is a different type.  All the tiles are initialized with ones.

    Parameters
    ----------
    dtype : np.number
        The data type of all the tiles except the hyd=0,ch=0,z=0 corner.
    corner_dtype
        The data type of the tile in the hyd=0,ch=0,z=0 corner.
    """
    def __init__(self, dtype: np.number, corner_dtype: np.number) -> None:
        self._dtype = dtype
        self._corner_dtype = corner_dtype

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        if round_label == 0 and ch_label == 0 and zplane_label == 0:
            dtype = self._corner_dtype
        else:
            dtype = self._dtype

        return OnesTilesByDtype(dtype)


def test_multiple_tiles_of_different_kind():
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=CornerDifferentDtype(np.uint32, np.float32),
    )
    with pytest.raises(TypeError):
        stack._ensure_data_loaded()


def test_multiple_tiles_of_same_dtype():
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=CornerDifferentDtype(np.uint32, np.uint32),
    )
    expected = np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.uint32)
    assert np.array_equal(stack.xarray, img_as_float32(expected))


def test_int_type_promotion():
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=CornerDifferentDtype(np.int32, np.int8),
    )
    with warnings.catch_warnings(record=True) as warnings_:
        stack._ensure_data_loaded()
        for warning in warnings_:
            if issubclass(warning.category, DataFormatWarning):
                break
        else:
            pytest.fail(f"No {DataFormatWarning.__name__} warning raised.")
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
    assert np.array_equal(stack.xarray, img_as_float32(expected))


def test_uint_type_promotion():
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=CornerDifferentDtype(np.uint32, np.uint8),
    )
    with warnings.catch_warnings(record=True) as warnings_:
        stack._ensure_data_loaded()
        for warning in warnings_:
            if issubclass(warning.category, DataFormatWarning):
                break
        else:
            pytest.fail(f"No {DataFormatWarning.__name__} warning raised.")
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
    assert np.array_equal(stack.xarray, img_as_float32(expected))


def test_float_type_demotion():
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=CornerDifferentDtype(np.float64, np.float32),
    )
    with warnings.catch_warnings(record=True) as warnings_:
        stack._ensure_data_loaded()
        for warning in warnings_:
            if issubclass(warning.category, DataFormatWarning):
                break
        else:
            pytest.fail(f"No {DataFormatWarning.__name__} warning raised.")
    expected = np.ones(
        (NUM_ROUND,
         NUM_CH,
         NUM_Z,
         HEIGHT,
         WIDTH), dtype=np.float64)
    assert np.array_equal(stack.xarray, expected)
