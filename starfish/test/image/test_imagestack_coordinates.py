from typing import Mapping, Tuple, Union

import numpy as np
from slicedimage import ImageFormat

from starfish.experiment.builder import FetchedTile, tile_fetcher_factory
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Coordinates, Indices, Number

NUM_ROUND = 8
NUM_CH = 1
NUM_Z = 1
HEIGHT = 10
WIDTH = 10


def round_to_x(r: int) -> Tuple[float, float]:
    return (r + 1) * 1000, (r + 1) * 100


def round_to_y(r: int) -> Tuple[float, float]:
    return (r + 1) * 10, (r + 1) * 0.1


def round_to_z(r: int) -> Tuple[float, float]:
    return (r + 1) * 0.01, (r + 1) * 0.001


class OffsettedTiles(FetchedTile):
    """Tiles that are physically offset based on round."""
    def __init__(self, fov: int, _round: int, ch: int, z: int) -> None:
        super().__init__()
        self._round = _round

    @property
    def shape(self) -> Tuple[int, ...]:
        return HEIGHT, WIDTH

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: round_to_x(self._round),
            Coordinates.Y: round_to_y(self._round),
            Coordinates.Z: round_to_z(self._round),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=np.float32)


def test_coordinates():
    """Set up an ImageStack with tiles that are offset based on round.  Verify that the coordinates
    retrieved match.
    """
    stack = ImageStack.synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=tile_fetcher_factory(
            OffsettedTiles,
            True,
        )
    )

    for _round in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }

                xmin, xmax = stack.coordinates(indices, Coordinates.X)
                ymin, ymax = stack.coordinates(indices, Coordinates.Y)
                zmin, zmax = stack.coordinates(indices, Coordinates.Z)

                expected_xmin, expected_xmax = round_to_x(_round)
                expected_ymin, expected_ymax = round_to_y(_round)
                expected_zmin, expected_zmax = round_to_z(_round)

                assert np.isclose(xmin, expected_xmin)
                assert np.isclose(xmax, expected_xmax)
                assert np.isclose(ymin, expected_ymin)
                assert np.isclose(ymax, expected_ymax)
                assert np.isclose(zmin, expected_zmin)
                assert np.isclose(zmax, expected_zmax)


class OffsettedScalarTiles(FetchedTile):
    """Tiles that are physically offset based on round, but only have a single scalar coordinate."""
    def __init__(self, fov: int, _round: int, ch: int, z: int) -> None:
        super().__init__()
        self._round = _round

    @property
    def shape(self) -> Tuple[int, ...]:
        return HEIGHT, WIDTH

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: round_to_x(self._round)[0],
            Coordinates.Y: round_to_y(self._round)[0],
            Coordinates.Z: round_to_z(self._round)[0],
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.ones((HEIGHT, WIDTH), dtype=np.float32)


def test_scalar_coordinates():
    """Set up an ImageStack where only a single scalar physical coordinate is provided per axis.
    Internally, this should be converted to a range where the two endpoints are identical to the
    physical coordinate provided.
    """
    stack = ImageStack.synthetic_stack(
        NUM_ROUND, NUM_CH, NUM_Z,
        HEIGHT, WIDTH,
        tile_fetcher=tile_fetcher_factory(
            OffsettedScalarTiles,
            True,
        )
    )

    for _round in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                indices = {
                    Indices.ROUND: _round,
                    Indices.CH: ch,
                    Indices.Z: z
                }

                xmin, xmax = stack.coordinates(indices, Coordinates.X)
                ymin, ymax = stack.coordinates(indices, Coordinates.Y)
                zmin, zmax = stack.coordinates(indices, Coordinates.Z)

                expected_x = round_to_x(_round)[0]
                expected_y = round_to_y(_round)[0]
                expected_z = round_to_z(_round)[0]

                assert np.isclose(xmin, expected_x)
                assert np.isclose(xmax, expected_x)
                assert np.isclose(ymin, expected_y)
                assert np.isclose(ymax, expected_y)
                assert np.isclose(zmin, expected_z)
                assert np.isclose(zmax, expected_z)
