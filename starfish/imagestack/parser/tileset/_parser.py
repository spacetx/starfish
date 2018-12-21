"""
This module parses and retains the extras metadata attached to TileSet extras.
"""
import warnings
from typing import Collection, MutableMapping, Tuple

from slicedimage import Tile, TileSet

from starfish.errors import DataFormatWarning
from starfish.imagestack.dataorder import AXES_DATA
from starfish.imagestack.parser import TileCollectionData, TileKey
from starfish.types import Indices


class TileSetData(TileCollectionData):
    """
    This class presents a simpler API for accessing a TileSet and its constituent tiles.
    """
    def __init__(self, tileset: TileSet) -> None:
        self.tiles: MutableMapping[TileKey, Tile] = dict()
        for tile in tileset.tiles():
            key = TileKey(
                round=tile.indices[Indices.ROUND],
                ch=tile.indices[Indices.CH],
                z=tile.indices.get(Indices.Z, 0))
            self.tiles[key] = tile
        self._extras = tileset.extras

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return self.tiles[tilekey].extras

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        return self.tiles.keys()

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        return self._extras

    def get_tile(self, r: int, ch: int, z: int) -> Tile:
        return self.tiles[TileKey(round=r, ch=ch, z=z)]


def parse_tileset(
        tileset: TileSet
) -> Tuple[Tuple[int, int], TileSetData]:
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

    # NOTE: (ttung) this is highly inefficient as it forces us to decode the tiles multiple times.
    # However, to simplify code review, I'm going to punt this until the next PR.

    tile_shape = tileset.default_tile_shape

    # Examine the tiles to figure out the right kind (int, float, etc.) and size.  We require
    # that all the tiles have the same kind of data type, but we do not require that they all
    # have the same size of data type. The # allocated array is the highest size we encounter.
    kind = None
    max_size = 0
    max_dtype = None
    for tile in tileset.tiles():
        dtype = tile.numpy_array.dtype
        if kind is None:
            kind = dtype.kind
        else:
            if kind != dtype.kind:
                raise TypeError("All tiles should have the same kind of dtype")
        if dtype.itemsize > max_size:
            max_size = dtype.itemsize
            max_dtype = dtype
        if tile_shape is None:
            tile_shape = tile.tile_shape
        elif tile.tile_shape is not None and tile_shape != tile.tile_shape:
            raise ValueError("Starfish does not support tiles that are not identical in shape")
    for tile in tileset.tiles():
        if max_size != tile.numpy_array.dtype.itemsize:
            warnings.warn(
                f"Tile "
                f"(R: {tile.indices[Indices.ROUND]} C: {tile.indices[Indices.CH]} "
                f"Z: {tile.indices[Indices.Z]}) has "
                f"dtype {tile.numpy_array.dtype}.  One or more tiles is of a larger dtype "
                f"{max_dtype}.",
                DataFormatWarning)

    return (
        tile_shape,
        tile_data,
    )


def _get_dimension_size(tileset: TileSet, dimension: Indices):
    axis_data = AXES_DATA[dimension]
    if dimension in tileset.dimensions or axis_data.required:
        return tileset.get_dimension_shape(dimension)
    return 1
