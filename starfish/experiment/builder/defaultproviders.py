"""
This module implements default providers of data to the experiment builders.
"""

from typing import Mapping, Tuple, Union

import numpy as np
from slicedimage import (
    ImageFormat,
)

from starfish.types import Coordinates, Number
from .providers import FetchedTile, TileFetcher


class RandomNoiseTile(FetchedTile):
    """
    This is a simple implementation of :class:`.FetchedImage` that simply regenerates random data
    for the image.
    """
    @property
    def shape(self) -> Tuple[int, ...]:
        return 1536, 1024

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    @property
    def tile_data(self) -> np.ndarray:
        return np.random.randint(0, 256, size=self.shape, dtype=np.uint8)


class RandomNoiseTileFetcher(TileFetcher):
    """
    This is a simple implementation of :class:`.ImageFetcher` that simply returns a
    :class:`.RandomNoiseImage` for every fov, hyb, ch, z combination.
    """
    def get_tile(self, fov: int, hyb: int, ch: int, z: int) -> FetchedTile:
        return RandomNoiseTile()
