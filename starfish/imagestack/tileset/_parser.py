"""
This module parses and retains the extras metadata attached to TileSet extras.
"""
import warnings
from typing import Collection, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
from slicedimage import Tile, TileSet

from starfish.errors import DataFormatWarning
from starfish.imagestack.dataorder import AXES_DATA
from starfish.types import Coordinates, Indices, Number
from ._key import TileKey


class _Expectations:
    """
    This class tracks all the tile shapes and the dtypes seen thus far during the decode of a
    :py:class:`slicedimage.TileSet`.  If the shapes are not all identical, it will trigger a
    ValueError.  If the kind of dtypes are not all identical, it will trigger a TypeError.
    Additionally, if the dtypes are all of the same kind, but not all of the same size, it will
    trigger a :py:class:`starfish.errors.DataFormatWarning`.
    """
    def __init__(self) -> None:
        self.tile_shape: Optional[Tuple[int, ...]] = None
        self.kind = None
        self.dtype_size = None

    def report_tile_shape(self, r: int, ch: int, z: int, tile_shape: Tuple[int, ...]) -> None:
        if self.tile_shape is not None and self.tile_shape != tile_shape:
            raise ValueError("Starfish does not support tiles that are not identical in shape")
        self.tile_shape = tile_shape

    def report_dtype(self, r: int, ch: int, z: int, dtype) -> None:
        if self.kind is not None and self.kind != dtype.kind:
            raise TypeError("All tiles should have the same kind of dtype")
        if self.dtype_size is not None and self.dtype_size != dtype.itemsize:
            warnings.warn(
                f"Tile (R: {r} C: {ch} Z: {z}) has dtype {dtype}, which is different from one or "
                f"more of the toher tiles.",
                DataFormatWarning)

        self.kind = dtype.kind
        self.dtype_size = dtype.itemsize


class _ProxyTile:
    """
    This wraps a :py:class:`slicedimage.Tile` and is duck-typed to behave like a read-only version
    of :py:class:`slicedimage.Tile`.  The key behavioral difference between this and
    :py:class:`slicedimage.Tile` is that this class does cache the image data upon load.  It is
    therefore incumbent on the consumers of these objects to discard them as soon as it is
    reasonable to do so to free up memory.
    """
    def __init__(
            self,
            inner_tile: Tile,
            expectations: _Expectations,
            r: int,
            ch: int,
            z: int,
    ) -> None:
        self._inner_tile = inner_tile
        self._expectations = expectations
        self._r = r
        self._ch = ch
        self._z = z
        self._numpy_array: Optional[np.ndarray] = None

    def _load(self):
        if self._numpy_array is not None:
            return
        self._numpy_array = self._inner_tile.numpy_array
        self._expectations.report_dtype(self._r, self._ch, self._z, self._numpy_array.dtype)

    @property
    def tile_shape(self):
        self._load()
        tile_shape = self._numpy_array.shape
        self._expectations.report_tile_shape(self._r, self._ch, self._z, tile_shape)
        return tile_shape

    @property
    def numpy_array(self):
        self._load()
        return self._numpy_array

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Tuple[Number, Number]]:
        return self._inner_tile.coordinates

    @property
    def indices(self) -> Mapping[str, int]:
        return self._inner_tile.indices


class TileSetData:
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
        self._expectations = _Expectations()

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

    def get_tile_by_key(self, tilekey: TileKey) -> _ProxyTile:
        return _ProxyTile(
            self.tiles[tilekey],
            self._expectations,
            tilekey.round, tilekey.ch, tilekey.z,
        )

    def get_tile(self, r: int, ch: int, z: int) -> _ProxyTile:
        return _ProxyTile(
            self.tiles[TileKey(round=r, ch=ch, z=z)],
            self._expectations,
            r, ch, z,
        )


def parse_tileset(
        tileset: TileSet
) -> Tuple[Mapping[Indices, int], Tuple[int, int], TileSetData]:
    """
    Parse a :py:class:`slicedimage.TileSet` for formatting into an
    :py:class:`starfish.imagestack.ImageStack`.

    Parameters:
    -----------
    tileset : TileSet
        The tileset to parse.

    Returns:
    --------
    Tuple[Mapping[Indices, int], Tuple[int, int], TileSetData] :
        A tuple consisting of the following:
            1. A mapping from :py:class:`starfish.types.Indices` to the size of that index.
            2. The (y, x) size of each tile.
            3. A :py:class:`starfish.imagestack.tileset.TileSetData` that can be queried to obtain
               the image data and extras metadata of each tile, as well as the extras metadata of
               the entire :py:class:`slicedimage.TileSet`.
    """
    num_rounds = _get_dimension_size(tileset, Indices.ROUND)
    num_chs = _get_dimension_size(tileset, Indices.CH)
    num_zlayers = _get_dimension_size(tileset, Indices.Z)
    tile_data = TileSetData(tileset)

    tile_shape = tileset.default_tile_shape

    # if we don't have the tile shape, then we peek at the first tile and get its shape.
    if tile_shape is None:
        tile_key = next(iter(tile_data.keys()))
        tile = tile_data.get_tile_by_key(tile_key)
        tile_shape = tile.tile_shape

    return (
        {
            Indices.ROUND: num_rounds,
            Indices.CH: num_chs,
            Indices.Z: num_zlayers,
        },
        tile_shape,
        tile_data,
    )


def _get_dimension_size(tileset: TileSet, dimension: Indices):
    axis_data = AXES_DATA[dimension]
    if dimension in tileset.dimensions or axis_data.required:
        return tileset.get_dimension_shape(dimension)
    return 1
