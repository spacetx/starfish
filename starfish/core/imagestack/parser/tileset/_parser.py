"""
This module parses and retains the extras metadata attached to TileSet extras.
"""
from typing import Collection, Mapping, MutableMapping, Set, Tuple

import numpy as np
from slicedimage import Tile, TileSet
from slicedimage._dimensions import DimensionNames

from starfish.core.imagestack.dataorder import AXES_DATA
from starfish.core.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.core.imagestack.physical_coordinates import _get_physical_coordinates_of_z_plane
from starfish.core.types import ArrayLike, Axes, Coordinates, Number


class SlicedImageTile(TileData):
    """
    This wraps a :py:class:`slicedimage.Tile`.  The difference between this and
    :py:class:`slicedimage.Tile` is that this class does cache the image data upon load.  It is
    therefore incumbent on the consumers of these objects to discard them as soon as it is
    reasonable to do so to free up memory.
    """
    def __init__(
            self,
            wrapped_tile: Tile,
            r: int,
            ch: int,
            zplane: int,
    ) -> None:
        self._wrapped_tile = wrapped_tile
        self._r = r
        self._ch = ch
        self._zplane = zplane
        self._numpy_array: np.ndarray = None

    def _load(self):
        if self._numpy_array is not None:
            return
        self._numpy_array = self._wrapped_tile.numpy_array

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return {
            Axes.Y: self._wrapped_tile.tile_shape[DimensionNames.Y],
            Axes.X: self._wrapped_tile.tile_shape[DimensionNames.X],
        }

    @property
    def numpy_array(self) -> np.ndarray:
        self._load()
        return self._numpy_array

    @property
    def coordinates(self) -> Mapping[Coordinates, ArrayLike[Number]]:
        xrange = self._wrapped_tile.coordinates[Coordinates.X]
        yrange = self._wrapped_tile.coordinates[Coordinates.Y]
        return_coords = {
            Coordinates.X: np.linspace(xrange[0], xrange[1], self.tile_shape[Axes.X]),
            Coordinates.Y: np.linspace(yrange[0], yrange[1], self.tile_shape[Axes.Y]),
        }

        if Coordinates.Z in self._wrapped_tile.coordinates:
            zrange = self._wrapped_tile.coordinates[Coordinates.Z]
            zplane_coord = _get_physical_coordinates_of_z_plane(zrange)
            return_coords[Coordinates.Z] = [zplane_coord]

        return return_coords

    @property
    def selector(self) -> Mapping[Axes, int]:
        return {
            Axes(axis_str): index
            for axis_str, index in self._wrapped_tile.indices.items()
        }


class TileSetData(TileCollectionData):
    """
    This class presents a simpler API for accessing a TileSet and its constituent tiles.
    """
    def __init__(self, tileset: TileSet) -> None:
        self._tile_shape = tileset.default_tile_shape

        self.tiles: MutableMapping[TileKey, Tile] = dict()
        for tile in tileset.tiles():
            key = TileKey(
                round=tile.indices[Axes.ROUND],
                ch=tile.indices[Axes.CH],
                zplane=tile.indices.get(Axes.ZPLANE, 0))
            self.tiles[key] = tile

            # if we don't have the tile shape, then we peek at the tile and get its shape.
            if self._tile_shape is None:
                self._tile_shape = tile.tile_shape

        self._extras = tileset.extras

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return self.tiles[tilekey].extras

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        return self.tiles.keys()

    @property
    def group_by(self) -> Set[Axes]:
        """Returns the axes to group by when we load the data."""
        return set()

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return self._tile_shape

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        return self._extras

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        return SlicedImageTile(
            self.tiles[tilekey],
            tilekey.round, tilekey.ch, tilekey.z,
        )

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        return SlicedImageTile(
            self.tiles[TileKey(round=r, ch=ch, zplane=z)],
            r, ch, z,
        )


def parse_tileset(
        tileset: TileSet
) -> Tuple[Mapping[Axes, int], TileCollectionData]:
    """
    Parse a :py:class:`slicedimage.TileSet` for formatting into an
    :py:class:`starfish.imagestack.ImageStack`.

    Parameters:
    -----------
    tileset : TileSet
        The tileset to parse.

    Returns:
    --------
    Tuple[Tuple[int, int], TileSetData] :
        A tuple consisting of the following:
            1. The (y, x) size of each tile.
            2. A :py:class:`starfish.imagestack.tileset.TileSetData` that can be queried to obtain
               the image data and extras metadata of each tile, as well as the extras metadata of
               the entire :py:class:`slicedimage.TileSet`.
    """
    tile_data = TileSetData(tileset)

    tile_shape = tileset.default_tile_shape

    # if we don't have the tile shape, then we peek at the first tile and get its shape.
    if tile_shape is None:
        tile_key = next(iter(tile_data.keys()))
        tile = tile_data.get_tile_by_key(tile_key)
        tile_shape = tile.tile_shape

    return (
        tile_shape,
        tile_data,
    )


def _get_dimension_size(tileset: TileSet, dimension: Axes):
    axis_data = AXES_DATA[dimension]
    if dimension in tileset.dimensions or axis_data.required:
        return tileset.get_dimension_shape(dimension)
    return 1
