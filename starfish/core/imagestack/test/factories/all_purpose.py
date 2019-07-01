from abc import ABCMeta
from typing import Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from starfish.core.experiment.builder import (
    build_image,
    FetchedTile,
    tile_fetcher_factory,
    TileFetcher,
)
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.core.types import Axes, Coordinates, CoordinateValue, Number


class LocationAwareFetchedTile(FetchedTile, metaclass=ABCMeta):
    """This is the base class for tiles that are aware of their location in the 5D tensor.
    """
    def __init__(
            self,
            # these are the arguments passed in as a result of tile_fetcher_factory's
            # pass_tile_indices parameter.
            fov_id: int, round_id: int, ch_id: int, zplane_id: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            fovs: Sequence[int], rounds: Sequence[int], chs: Sequence[int], zplanes: Sequence[int],
            tile_height: int, tile_width: int,
    ) -> None:
        super().__init__()
        self.fov_id = fov_id
        self.round_id = round_id
        self.ch_id = ch_id
        self.zplane_id = zplane_id
        self.fovs = fovs
        self.rounds = rounds
        self.chs = chs
        self.zplanes = zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width


def _apply_coords_range_fetcher(
        backing_tile_fetcher: TileFetcher,
        zplanes: Sequence[int],
        fov_to_xrange: Mapping[int, Tuple[Number, Number]],
        fov_to_yrange: Mapping[int, Tuple[Number, Number]],
        zrange: Tuple[Number, Number],
) -> TileFetcher:
    """Given a :py:class:`TileFetcher`, intercept all the returned :py:class:`FetchedTile` instances
    and replace the coordinates.  The range for the x and the y coordinates should be fetched from
    `fov_to_xrange` and `fov_to_yrange`, respectively, using the fov id.  The z coordinates are
    uniform across all fields of view."""
    class ModifiedTile(FetchedTile):
        def __init__(self, backing_tile: FetchedTile, fov: int, zplane: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backing_tile = backing_tile
            self.fov = fov
            self.zplane = zplane

        @property
        def shape(self) -> Mapping[Axes, int]:
            return self.backing_tile.shape

        @property
        def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
            zplane_offset = zplanes.index(self.zplane)
            zplane_coords = np.linspace(zrange[0], zrange[1], len(zplanes))

            return {
                Coordinates.X: fov_to_xrange[self.fov],
                Coordinates.Y: fov_to_yrange[self.fov],
                Coordinates.Z: zplane_coords[zplane_offset],
            }

        def tile_data(self) -> np.ndarray:
            return self.backing_tile.tile_data()

    class ModifiedTileFetcher(TileFetcher):
        def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
            original_fetched_tile = backing_tile_fetcher.get_tile(fov, r, ch, z)
            return ModifiedTile(original_fetched_tile, fov, z)

    return ModifiedTileFetcher()


def imagestack_factory(
        fetched_tile_cls: Type[LocationAwareFetchedTile],
        round_labels: Sequence[int],
        ch_labels: Sequence[int],
        zplane_labels: Sequence[int],
        tile_height: int,
        tile_width: int,
        xrange: Tuple[Number, Number],
        yrange: Tuple[Number, Number],
        zrange: Tuple[Number, Number],
        crop_parameters: Optional[CropParameters] = None) -> ImageStack:
    """Given a type that implements the :py:class:`LocationAwareFetchedTile` contract, produce an
    imagestack with those tiles, and apply coordinates such that the 5D tensor has coordinates
    that range from `xrange[0]:xrange[1]`, `yrange[0]:yrange[1]`, `zrange[0]:zrange[1]`.

    Parameters
    ----------
    fetched_tile_cls : Type[LocationAwareFetchedTile]
        The class of the FetchedTile.
    round_labels : Sequence[int]
        Labels for the rounds.
    ch_labels : Sequence[int]
        Labels for the channels.
    zplane_labels : Sequence[int]
        Labels for the zplanes.
    tile_height : int
        Height of each tile, in pixels.
    tile_width : int
        Width of each tile, in pixels.
    xrange : Tuple[Number, Number]
        The starting and ending x physical coordinates for the tile.
    yrange : Tuple[Number, Number]
        The starting and ending y physical coordinates for the tile.
    zrange : Tuple[Number, Number]
        The starting and ending z physical coordinates for the tile.
    crop_parameters : Optional[CropParameters]
        The crop parameters to apply during ImageStack construction.
    """
    original_tile_fetcher = tile_fetcher_factory(
        fetched_tile_cls, True,
        range(1), round_labels, ch_labels, zplane_labels,
        tile_height, tile_width,
    )
    modified_tile_fetcher = _apply_coords_range_fetcher(
        original_tile_fetcher, zplane_labels, {0: xrange}, {0: yrange}, zrange)

    collection = build_image(
        range(1),
        round_labels,
        ch_labels,
        zplane_labels,
        modified_tile_fetcher,
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset, crop_parameters)
