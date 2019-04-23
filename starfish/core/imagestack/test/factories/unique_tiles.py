from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from skimage import img_as_float32
from slicedimage import ImageFormat

from starfish.core.experiment.builder import build_image, FetchedTile, tile_fetcher_factory
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.core.types import Axes, Coordinates, Number

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


class UniqueTiles(FetchedTile):
    """Tiles where the pixel values are unique per round/ch/z."""
    def __init__(
            self,
            # these are the arguments passed in as a result of tile_fetcher_factory's
            # pass_tile_indices parameter.
            fov: int, _round: int, ch: int, zplane: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            num_rounds: int, num_chs: int, num_zplanes: int, tile_height: int, tile_width: int
    ) -> None:
        super().__init__()
        self._round = _round
        self._ch = ch
        self._zplane = zplane
        self.num_rounds = num_rounds
        self.num_chs = num_chs
        self.num_zplanes = num_zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self.tile_height, Axes.X: self.tile_width}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: X_COORDS,
            Coordinates.Y: Y_COORDS,
            Coordinates.Z: Z_COORDS,
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return unique_data(
            self._round, self._ch, self._zplane,
            self.num_rounds, self.num_chs, self.num_zplanes,
            self.tile_height, self.tile_width,
        )


def unique_tiles_imagestack(
        round_labels: Sequence[int],
        ch_labels: Sequence[int],
        z_labels: Sequence[int],
        tile_height: int,
        tile_width: int,
        crop_parameters: Optional[CropParameters] = None) -> ImageStack:
    """Build an imagestack with unique values per tile.
    """
    collection = build_image(
        range(1),
        round_labels,
        ch_labels,
        z_labels,
        tile_fetcher_factory(
            UniqueTiles, True,
            len(round_labels), len(ch_labels), len(z_labels),
            tile_height, tile_width,
        ),
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset, crop_parameters)
