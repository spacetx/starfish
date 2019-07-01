from abc import ABCMeta
from typing import Mapping

import numpy as np
from skimage import img_as_float32
from slicedimage import ImageFormat

from starfish.core.types import Axes
from .all_purpose import LocationAwareFetchedTile

X_COORDS = 0.01, 0.1
Y_COORDS = 0.001, 0.01
Z_COORDS = 0.0001, 0.001


def unique_data(
        fov_id: int, round_label_offset: int, ch_label_offset: int, zplane_label_offset: int,
        num_fovs: int, num_rounds: int, num_chs: int, num_zplanes: int,
        tile_height: int, tile_width: int,
) -> np.ndarray:
    """Return the data for a given tile."""
    result = np.empty((tile_height, tile_width), dtype=np.uint32)
    for row in range(tile_height):
        base_val = tile_width * (
            row + tile_height * (
                zplane_label_offset + num_zplanes * (
                    ch_label_offset + num_chs * (
                        round_label_offset + num_rounds * (
                            fov_id)))))

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
            self.fov_id,
            self.rounds.index(self.round_label),
            self.chs.index(self.ch_label),
            self.zplanes.index(self.zplane_label),
            len(self.fovs),
            len(self.rounds),
            len(self.chs),
            len(self.zplanes),
            self.tile_height,
            self.tile_width)
