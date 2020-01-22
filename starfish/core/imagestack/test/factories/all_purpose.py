from typing import Callable, Collection, Mapping, Optional, Sequence, Tuple, Type

from starfish.core.experiment.builder.builder import TileIdentifier
from starfish.core.experiment.builder.test.factories.all_purpose import (
    collection_factory,
    LocationAwareFetchedTile,
)
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.core.types import Coordinates, CoordinateValue, Number


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
    tile_identifiers: Collection[TileIdentifier] = [
        TileIdentifier(0, round_label, ch_label, zplane_label)
        for round_label in round_labels
        for ch_label in ch_labels
        for zplane_label in zplane_labels
    ]

    def make_tile_coordinate_callback(
            all_zplane_labels: Sequence[int]
    ) -> Callable[[TileIdentifier], Mapping[Coordinates, CoordinateValue]]:
        def tile_coordinate_callback(
                tile_identifier: TileIdentifier
        ) -> Mapping[Coordinates, CoordinateValue]:
            zplane_offset = all_zplane_labels.index(tile_identifier.zplane_label)
            return {
                Coordinates.X: xrange,
                Coordinates.Y: yrange,
                Coordinates.Z: zrange[zplane_offset],
            }

        return tile_coordinate_callback

    collection = collection_factory(
        fetched_tile_cls,
        tile_identifiers,
        make_tile_coordinate_callback(
            sorted(set(tile_identifier.zplane_label for tile_identifier in tile_identifiers))),
        tile_height,
        tile_width,
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset, crop_parameters)
