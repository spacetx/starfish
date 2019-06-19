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
            fov: int, _round: int, ch: int, zplane: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            rounds: Sequence[int], chs: Sequence[int], zplanes: Sequence[int],
            tile_height: int, tile_width: int,
    ) -> None:
        super().__init__()
        self.round = _round
        self.ch = ch
        self.zplane = zplane
        self.rounds = rounds
        self.chs = chs
        self.zplanes = zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width


def _apply_coords_range_fetcher(
        backing_tile_fetcher: TileFetcher,
        zplanes: Sequence[int],
        xrange: Tuple[Number, Number],
        yrange: Tuple[Number, Number],
        zrange: Tuple[Number, Number],
) -> TileFetcher:
    """Given a :py:class:`TileFetcher`, intercept all the returned :py:class:`FetchedTile` instances
    and replace the coordinates such that the resulting tensor has coordinates that range from
    `xrange[0]:xrange[1]`, `yrange[0]:yrange[1]`, `zrange[0]:zrange[1]` """
    class ModifiedTile(FetchedTile):
        def __init__(self, backing_tile: FetchedTile, zplane: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backing_tile = backing_tile
            self.zplane = zplane

        @property
        def shape(self) -> Mapping[Axes, int]:
            return self.backing_tile.shape

        @property
        def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
            zplane_offset = zplanes.index(self.zplane)
            zplane_coords = np.linspace(zrange[0], zrange[1], len(zplanes))

            return {
                Coordinates.X: xrange,
                Coordinates.Y: yrange,
                Coordinates.Z: zplane_coords[zplane_offset],
            }

        def tile_data(self) -> np.ndarray:
            return self.backing_tile.tile_data()

    class ModifiedTileFetcher(TileFetcher):
        def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
            original_fetched_tile = backing_tile_fetcher.get_tile(fov, r, ch, z)
            return ModifiedTile(original_fetched_tile, z)

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
        round_labels, ch_labels, zplane_labels,
        tile_height, tile_width,
    )
    modified_tile_fetcher = _apply_coords_range_fetcher(
        original_tile_fetcher, zplane_labels, xrange, yrange, zrange)

    collection = build_image(
        range(1),
        round_labels,
        ch_labels,
        zplane_labels,
        modified_tile_fetcher,
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset, crop_parameters)
