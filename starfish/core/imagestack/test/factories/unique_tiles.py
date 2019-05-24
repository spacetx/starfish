from abc import ABCMeta
from typing import Mapping, Optional, Sequence

import numpy as np
from skimage import img_as_float32
from slicedimage import ImageFormat

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.core.types import Axes
from .all_purpose import imagestack_factory, LocationAwareFetchedTile

X_COORDS = 0.01, 0.1
Y_COORDS = 0.001, 0.01
Z_COORDS = 0.0001, 0.001


def unique_data(
        round_: int, ch: int, z: int,
        num_rounds: int, num_chs: int, num_zplanes: int,
        tile_height: int, tile_width: int,
) -> np.ndarray:
    """Return the data for a given tile."""
    result = np.empty((tile_height, tile_width), dtype=np.uint32)
    for row in range(tile_height):
        base_val = tile_width * (
            row + tile_height * (
                z + num_zplanes * (
                    ch + num_chs * (
                        round_))))

        result[row:] = np.linspace(base_val, base_val + tile_width, tile_width, False)
    return img_as_float32(result)


class UniqueTiles(LocationAwareFetchedTile, metaclass=ABCMeta):
    """Tiles where the pixel values are unique per round/ch/z."""
    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self.tile_height, Axes.X: self.tile_width}

    def tile_data(self) -> np.ndarray:
        """Return the data for a given tile."""
        return unique_data(
            self.rounds.index(self.round),
            self.chs.index(self.ch),
            self.zplanes.index(self.zplane),
            len(self.rounds),
            len(self.chs),
            len(self.zplanes),
            self.tile_height,
            self.tile_width,
        )


def unique_tiles_imagestack(
        round_labels: Sequence[int],
        ch_labels: Sequence[int],
        zplane_labels: Sequence[int],
        tile_height: int,
        tile_width: int,
        crop_parameters: Optional[CropParameters] = None) -> ImageStack:
    """Build an imagestack with unique values per tile.
    """
    return imagestack_factory(
        UniqueTiles,
        round_labels,
        ch_labels,
        zplane_labels,
        tile_height,
        tile_width,
        X_COORDS,
        Y_COORDS,
        Z_COORDS,
        crop_parameters,
    )
