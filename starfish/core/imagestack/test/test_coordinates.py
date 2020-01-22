from typing import Mapping, Tuple, Union

import numpy as np
from slicedimage import ImageFormat

from starfish.core.experiment.builder.builder import tile_fetcher_factory
from starfish.core.experiment.builder.providers import FetchedTile
from starfish.types import Axes, Coordinates, CoordinateValue
from .factories import synthetic_stack
from .imagestack_test_utils import verify_physical_coordinates
from ..physical_coordinates import _get_physical_coordinates_of_z_plane

NUM_ROUND = 8
NUM_CH = 1
NUM_Z = 3
HEIGHT = 10
WIDTH = 10


X_COORDS = 100, 1000
Y_COORDS = .1, 10


def zplane_to_z(z: int) -> Tuple[float, float]:
    return (z + 1) * 0.01, (z + 1) * 0.001


def round_to_x(r: int) -> Tuple[float, float]:
    return (r + 1) * 0.001, (r + 1) * 0.0001


def round_to_y(r: int) -> Tuple[float, float]:
    return (r + 1) * 0.1, (r + 1) * 1


class AlignedTiles(FetchedTile):
    """Tiles that are physically offset based on round."""

    def __init__(self, fov: int, _round: int, ch: int, z: int) -> None:
        super().__init__()
        self._round = _round
        self._zplane = z

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: HEIGHT, Axes.X: WIDTH}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: X_COORDS,
            Coordinates.Y: Y_COORDS,
            Coordinates.Z: zplane_to_z(self._zplane),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=np.float32)


def test_coordinates():
    """Set up an ImageStack with tiles that are aligned.  Verify that the coordinates
    retrieved match.
    """
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=tile_fetcher_factory(
            AlignedTiles,
            True,
        )
    )
    for selectors in stack._iter_axes({Axes.ZPLANE}):
        verify_physical_coordinates(stack, X_COORDS, Y_COORDS,
                                    _get_physical_coordinates_of_z_plane(
                                        zplane_to_z(selectors[Axes.ZPLANE])),
                                    selectors[Axes.ZPLANE])


class ScalarTiles(FetchedTile):
    """Tiles that have a single scalar coordinate."""
    def __init__(self, fov: int, _round: int, ch: int, z: int) -> None:
        super().__init__()
        self._round = _round
        self._zplane = z

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: HEIGHT, Axes.X: WIDTH}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: X_COORDS[0],
            Coordinates.Y: Y_COORDS[0],
            Coordinates.Z: zplane_to_z(self._zplane)[0],
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=np.float32)


class OffsettedTiles(FetchedTile):
    """Tiles that are physically offset based on round."""
    def __init__(self, fov: int, round_label: int, ch_label: int, z_label: int) -> None:
        super().__init__()
        self.round_label = round_label

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: HEIGHT, Axes.X: WIDTH}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: round_to_x(self.round_label),
            Coordinates.Y: round_to_y(self.round_label),
            Coordinates.Z: zplane_to_z(self.round_label),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=np.float32)


def test_unaligned_tiles():
    """Test that imagestack error is thrown when constructed with unaligned tiles"""

    try:
        synthetic_stack(
            NUM_ROUND, NUM_CH, NUM_Z,
            HEIGHT, WIDTH,
            tile_fetcher=tile_fetcher_factory(
                OffsettedTiles,
                True,
            )
        )
    except ValueError as e:
        # Assert value error is thrown with right message
        assert e.args[0] == "Tiles must be aligned"


def test_scalar_coordinates():
    """Set up an ImageStack where only a single scalar physical coordinate is provided per axis.
    Internally, this should be converted to a range where the two endpoints are identical to the
    physical coordinate provided.
    """
    stack = synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=tile_fetcher_factory(
            ScalarTiles,
            True,
        )
    )

    expected_x = X_COORDS[0]
    expected_y = Y_COORDS[0]

    for selectors in stack._iter_axes({Axes.ZPLANE}):
        expected_z = zplane_to_z(selectors[Axes.ZPLANE])[0]
        verify_physical_coordinates(stack,
                                    (expected_x, expected_x),
                                    (expected_y, expected_y),
                                    _get_physical_coordinates_of_z_plane((expected_z, expected_z)),
                                    selectors[Axes.ZPLANE])
