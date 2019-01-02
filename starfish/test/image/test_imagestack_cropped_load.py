"""
These tests center around creating an ImageStack but selectively loading data from the original
TileSet.
"""
from typing import Mapping, Optional, Tuple, Union

import numpy as np
from skimage import img_as_float32
from slicedimage import ImageFormat

from starfish.experiment.builder import build_image, FetchedTile, tile_fetcher_factory
from starfish.imagestack.imagestack import ImageStack
from starfish.imagestack.parser.crop import CropParameters
from starfish.imagestack.physical_coordinate_calculator import recalculate_physical_coordinate_range
from starfish.types import Coordinates, Indices, Number
from .imagestack_test_utils import verify_physical_coordinates, verify_stack_data

NUM_ROUND = 3
NUM_CH = 4
NUM_Z = 2

ROUND_LABELS = list(range(NUM_ROUND))
CH_LABELS = list(range(NUM_CH))
Z_LABELS = list(range(NUM_Z))
HEIGHT = 40
WIDTH = 60


def data(round_: int, ch: int, z: int) -> np.ndarray:
    """Return the data for a given tile."""
    result = np.empty((HEIGHT, WIDTH), dtype=np.uint32)
    for row in range(HEIGHT):
        base_val = ((((((round_ * NUM_CH) + ch) * NUM_Z) + z) * HEIGHT) + row) * WIDTH

        result[row:] = np.linspace(base_val, base_val + WIDTH, WIDTH, False)
    return img_as_float32(result)


def x_coordinates(round_: int, ch: int) -> Tuple[float, float]:
    """Return the expected physical x coordinate value for a given round/ch tuple.  Note that in
    real life, physical coordinates are not expected to vary for different ch values.  However, for
    completeness of the tests, we are pretending they will."""
    return min(round_, ch) * 0.01, max(round_, ch) * 0.01


def y_coordinates(round_: int, ch: int) -> Tuple[float, float]:
    """Return the expected physical y coordinate value for a given round/ch tuple.  Note that in
    real life, physical coordinates are not expected to vary for different ch values.  However, for
    completeness of the tests, we are pretending they will."""
    return min(round_, ch) * 0.001, max(round_, ch) * 0.001


def z_coordinates(z: int) -> Tuple[float, float]:
    """Return the expected physical z coordinate value for a given zlayer index."""
    return z * 0.0001, (z + 1) * 0.0001


class UniqueTiles(FetchedTile):
    """Tiles where the pixel values are unique per round/ch/z."""
    def __init__(self, fov: int, _round: int, ch: int, z: int) -> None:
        super().__init__()
        self._round = _round
        self._ch = ch
        self._z = z

    @property
    def shape(self) -> Tuple[int, ...]:
        return HEIGHT, WIDTH

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: x_coordinates(self._round, self._ch),
            Coordinates.Y: y_coordinates(self._round, self._ch),
            Coordinates.Z: z_coordinates(self._z),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return data(self._round, self._ch, self._z)


def setup_imagestack(crop_parameters: Optional[CropParameters]) -> ImageStack:
    """Build an imagestack with labeled indices (i.e., indices that do not start at 0 or are not
    sequential non-negative integers).
    """
    collection = build_image(
        range(1),
        ROUND_LABELS,
        CH_LABELS,
        Z_LABELS,
        tile_fetcher_factory(UniqueTiles, True),
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset, crop_parameters)


def test_crop_rcz():
    """Build an imagestack that contains a crop in r/c/z.  Verify that the appropriate tiles are
    loaded.
    """
    rounds = [1]
    chs = [2, 3]

    crop_parameters = CropParameters(
        permitted_rounds=rounds,
        permitted_chs=chs,
    )
    stack = setup_imagestack(crop_parameters)

    assert stack.index_labels(Indices.ROUND) == rounds
    assert stack.index_labels(Indices.CH) == chs
    assert stack.index_labels(Indices.Z) == Z_LABELS

    for round_ in stack.index_labels(Indices.ROUND):
        for ch in stack.index_labels(Indices.CH):
            for zlayer in stack.index_labels(Indices.Z):
                expected_data = data(round_, ch, zlayer)

                verify_stack_data(
                    stack,
                    {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: zlayer},
                    expected_data,
                )

                verify_physical_coordinates(
                    stack,
                    {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: zlayer},
                    x_coordinates(round_, ch),
                    y_coordinates(round_, ch),
                    z_coordinates(zlayer),
                )


def test_crop_xy():
    """Build an imagestack that contains a crop in x/y.  Verify that the data is sliced correctly.
    """
    X_SLICE = (10, 30)
    Y_SLICE = (15, 40)
    crop_parameters = CropParameters(
        x_slice=slice(*X_SLICE),
        y_slice=slice(*Y_SLICE),
    )
    stack = setup_imagestack(crop_parameters)

    assert stack.index_labels(Indices.ROUND) == ROUND_LABELS
    assert stack.index_labels(Indices.CH) == CH_LABELS
    assert stack.index_labels(Indices.Z) == Z_LABELS

    assert stack.raw_shape[3] == Y_SLICE[1] - Y_SLICE[0]
    assert stack.raw_shape[4] == X_SLICE[1] - X_SLICE[0]

    for round_ in stack.index_labels(Indices.ROUND):
        for ch in stack.index_labels(Indices.CH):
            for zlayer in stack.index_labels(Indices.Z):
                expected_data = data(round_, ch, zlayer)
                expected_data = expected_data[Y_SLICE[0]:Y_SLICE[1], X_SLICE[0]:X_SLICE[1]]

                verify_stack_data(
                    stack,
                    {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: zlayer},
                    expected_data,
                )

                # the coordinates should be rescaled.  verify that the coordinates on the ImageStack
                # are also rescaled.
                original_x_coordinates = x_coordinates(round_, ch)
                expected_x_coordinates = recalculate_physical_coordinate_range(
                    original_x_coordinates[0], original_x_coordinates[1],
                    WIDTH,
                    slice(*X_SLICE),
                )

                original_y_coordinates = y_coordinates(round_, ch)
                expected_y_coordinates = recalculate_physical_coordinate_range(
                    original_y_coordinates[0], original_y_coordinates[1],
                    HEIGHT,
                    slice(*Y_SLICE),
                )

                verify_physical_coordinates(
                    stack,
                    {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: zlayer},
                    expected_x_coordinates,
                    expected_y_coordinates,
                    z_coordinates(zlayer),
                )
