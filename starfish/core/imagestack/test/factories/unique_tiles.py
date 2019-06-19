from typing import Optional, Sequence

from starfish.core.experiment.builder.test.factories.unique_tiles import UniqueTiles
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from .all_purpose import imagestack_factory

X_COORDS = 0.01, 0.1
Y_COORDS = 0.001, 0.01
Z_COORDS = 0.0001, 0.001


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
