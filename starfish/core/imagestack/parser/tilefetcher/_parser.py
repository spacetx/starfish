"""
This module wraps a TileFetcher to provide the data to instantiate an ImageStack.
"""
from typing import Collection, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from starfish.core.experiment.builder.orderediterator import join_axes_labels, ordered_iterator
from starfish.core.experiment.builder.providers import FetchedTile, TileFetcher
from starfish.core.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.core.imagestack.physical_coordinates import _get_physical_coordinates_of_z_plane
from starfish.core.types import ArrayLike, Axes, Coordinates, Number


class TileFetcherImageTile(TileData):
    """
    This wraps a :py:class:`TileFetcher`.
    """
    def __init__(
            self,
            tile: FetchedTile,
            r: int,
            ch: int,
            zplane: int,
    ) -> None:
        self._tile = tile
        self._r = r
        self._ch = ch
        self._zplane = zplane

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return self._tile.shape

    @property
    def numpy_array(self) -> np.ndarray:
        return self._tile.tile_data()

    @property
    def coordinates(self) -> Mapping[Coordinates, ArrayLike[Number]]:
        return_coords: MutableMapping[Coordinates, ArrayLike[Number]] = dict()
        for axis, coord in ((Axes.X, Coordinates.X), (Axes.Y, Coordinates.Y)):
            coordinate_range = self._tile.coordinates[coord]
            if not isinstance(coordinate_range, tuple):
                raise ValueError("x-y coordinates for a tile must be a range expressed as a tuple.")
            return_coords[coord] = np.linspace(
                coordinate_range[0], coordinate_range[1], self.tile_shape[axis])

        if Coordinates.Z in self._tile.coordinates:
            zrange = self._tile.coordinates[Coordinates.Z]
            if isinstance(zrange, tuple):
                zplane_coord = _get_physical_coordinates_of_z_plane(zrange)
                return_coords[Coordinates.Z] = [zplane_coord]
            else:
                return_coords[Coordinates.Z] = zrange

        return return_coords

    @property
    def selector(self) -> Mapping[Axes, int]:
        return {
            Axes.ROUND: self._r,
            Axes.CH: self._ch,
            Axes.ZPLANE: self._zplane,
        }


class TileFetcherData(TileCollectionData):
    """
    This class wraps a TileFetcher along with the axes labels to provide a 5D tensor suitable for
    loading as an ImageStack.
    """
    def __init__(
            self,
            tile_fetcher: TileFetcher,
            tile_shape: Mapping[Axes, int],
            fov: int,
            rounds: Sequence[int],
            chs: Sequence[int],
            zplanes: Sequence[int],
            axes_order: Optional[Sequence[Axes]] = None,
    ) -> None:
        self._tile_fetcher = tile_fetcher
        self._tile_shape = tile_shape
        self._fov = fov
        self._rounds = rounds
        self._chs = chs
        self._zplanes = zplanes
        if axes_order is None:
            axes_order = Axes.ZPLANE, Axes.ROUND, Axes.CH
        self._axes_order = axes_order

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return {}

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        axes_sizes = join_axes_labels(
            self._axes_order, rounds=self._rounds, chs=self._chs, zplanes=self._zplanes)
        return [
            TileKey(round=selector[Axes.ROUND], ch=selector[Axes.CH], zplane=selector[Axes.ZPLANE])
            for selector in ordered_iterator(axes_sizes)
        ]

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return self._tile_shape

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        return {}

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        return TileFetcherImageTile(
            self._tile_fetcher.get_tile(self._fov, tilekey.round, tilekey.ch, tilekey.z),
            tilekey.round,
            tilekey.ch,
            tilekey.z,
        )

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        return TileFetcherImageTile(
            self._tile_fetcher.get_tile(self._fov, r, ch, z),
            r,
            ch,
            z,
        )
