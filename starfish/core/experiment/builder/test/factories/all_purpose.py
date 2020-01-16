from abc import ABCMeta
from typing import Callable, cast, Collection, Mapping, Sequence, Type, Union

import numpy as np
import slicedimage

from starfish.core.types import Axes, Coordinates, CoordinateValue
from ...builder import (
    build_irregular_image,
    tile_fetcher_factory,
    TileIdentifier,
)
from ...providers import (
    FetchedTile,
    TileFetcher,
)


class LocationAwareFetchedTile(FetchedTile, metaclass=ABCMeta):
    """This is the base class for tiles that are aware of their location in the 5D tensor.
    """
    def __init__(
            self,
            # these are the arguments passed in as a result of tile_fetcher_factory's
            # pass_tile_indices parameter.
            fov_id: int, round_label: int, ch_label: int, zplane_label: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            fovs: Sequence[int], rounds: Sequence[int], chs: Sequence[int], zplanes: Sequence[int],
            tile_height: int, tile_width: int,
    ) -> None:
        super().__init__()
        self.fov_id = fov_id
        self.round_label = round_label
        self.ch_label = ch_label
        self.zplane_label = zplane_label
        self.fovs = fovs
        self.rounds = rounds
        self.chs = chs
        self.zplanes = zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width


def _apply_coords_range_fetcher(
        backing_tile_fetcher: TileFetcher,
        tile_coordinates_callback: Callable[
            [TileIdentifier], Mapping[Coordinates, CoordinateValue]],
) -> TileFetcher:
    """Given a :py:class:`TileFetcher`, intercept all the returned :py:class:`FetchedTile` instances
    and replace the coordinates using the coordinates from `tile_coordinates_callback`."""
    class ModifiedTile(FetchedTile):
        def __init__(
                self,
                backing_tile: FetchedTile,
                tile_identifier: TileIdentifier,
                *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.backing_tile = backing_tile
            self.tile_identifier = tile_identifier

        @property
        def shape(self) -> Mapping[Axes, int]:
            return self.backing_tile.shape

        @property
        def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
            return cast(
                Mapping[Union[str, Coordinates], CoordinateValue],
                tile_coordinates_callback(self.tile_identifier))

        def tile_data(self) -> np.ndarray:
            return self.backing_tile.tile_data()

    class ModifiedTileFetcher(TileFetcher):
        def get_tile(
                self, fov_id: int, round_label: int, ch_label: int, zplane_label: int,
        ) -> FetchedTile:
            original_fetched_tile = backing_tile_fetcher.get_tile(
                fov_id, round_label, ch_label, zplane_label)
            tile_identifier = TileIdentifier(fov_id, round_label, ch_label, zplane_label)
            return ModifiedTile(original_fetched_tile, tile_identifier)

    return ModifiedTileFetcher()


def collection_factory(
        fetched_tile_cls: Type[LocationAwareFetchedTile],
        tile_identifiers: Collection[TileIdentifier],
        tile_coordinates_callback: Callable[
            [TileIdentifier], Mapping[Coordinates, CoordinateValue]],
        tile_height: int,
        tile_width: int,
) -> slicedimage.Collection:
    """Given a type that implements the :py:class:`LocationAwareFetchedTile` contract, produce a
    slicedimage Collection with the tiles in `tile_identifiers`.  For a given tile_identifier,
    retrieve the coordinates by invoking the callback `tile_coordinates_callback`.

    Parameters
    ----------
    fetched_tile_cls : Type[LocationAwareFetchedTile]
        The class of the FetchedTile.
    tile_identifiers : Collection[TileIdentifier]
        TileIdentifiers for each of the tiles in the collection.
    tile_coordinates_callback : Callable[[TileIdentifier], Mapping[Coordinates, CoordinatesValue]]
        A callable that returns the coordinates for a given tile's TileIdentifier.
    tile_height : int
        Height of each tile, in pixels.
    tile_width : int
        Width of each tile, in pixels.
    """
    all_fov_ids = sorted(set(
        tile_identifier.fov_id for tile_identifier in tile_identifiers))
    all_round_labels = sorted(set(
        tile_identifier.round_label for tile_identifier in tile_identifiers))
    all_ch_labels = sorted(set(
        tile_identifier.ch_label for tile_identifier in tile_identifiers))
    all_zplane_labels = sorted(set(
        tile_identifier.zplane_label for tile_identifier in tile_identifiers))

    original_tile_fetcher = tile_fetcher_factory(
        fetched_tile_cls, True,
        all_fov_ids, all_round_labels, all_ch_labels, all_zplane_labels,
        tile_height, tile_width,
    )
    modified_tile_fetcher = _apply_coords_range_fetcher(
        original_tile_fetcher, tile_coordinates_callback)

    return build_irregular_image(
        tile_identifiers,
        modified_tile_fetcher,
        default_shape={Axes.Y: tile_height, Axes.X: tile_width}
    )
