"""
This module parses and retains the extras metadata attached to TileSet extras.
"""
import warnings
from typing import Collection, Mapping, MutableMapping, Optional, Tuple

import numpy as np
from slicedimage import Tile, TileSet

from starfish.errors import DataFormatWarning
from starfish.imagestack.dataorder import AXES_DATA
from starfish.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.types import Axes, Coordinates, Number


class _TileSetConsistencyDetector:
    """
    This class tracks all the tile shapes and the dtypes seen thus far during the decode of a
    :py:class:`slicedimage.TileSet`.  If the shapes are not all identical, it will trigger a
    ValueError.  If the kind of dtypes are not all identical, it will trigger a TypeError.
    Additionally, if the dtypes are all of the same kind, but not all of the same size, it will
    trigger a :py:class:`starfish.errors.DataFormatWarning`.
    """
    def __init__(self) -> None:
        self.tile_shape: Optional[Mapping[Axes, int]] = None
        self.kind = None
        self.dtype_size = None

    def report_tile_shape(
            self,
            r: int, ch: int, zplane: int,
            tile_shape: Mapping[Axes, int]) -> None:
        """As each tile is parsed, the tile shape should be reported via this method.  If an
        inconsistency is detected, a ValueError exception will be raised.

        Parameters
        ----------
        r, ch, zplane : int
            The indices of the tile, which is used to generate the exception text.
        tile_shape : Mapping[Axes, int]
            The shape of the tile, mapping from Axes to its size.
        """
        if self.tile_shape is not None and self.tile_shape != tile_shape:
            raise ValueError(
                f"Tile (R: {r} C: {ch} Z: {zplane}) has shape {tile_shape}, which is different "
                f"from one or more of the other tiles.  Starfish does not support tiles that are "
                f"not identical in shape."
            )
        self.tile_shape = tile_shape

    def report_dtype(self, r: int, ch: int, zplane: int, dtype) -> None:
        """As each tile is parsed, the tile data type should be reported via this method.  If an
        inconsistency in the data type kind is is detected, a TypeError exception will be raised.
        If the data type are of the same kind but not the same size, it will trigger a
        :py:class:`starfish.errors.DataFormatWarning`

        Parameters
        ----------
        r, ch, zplane : int
            The indices of the tile, which is used to generate the exception text.
        dtype :
            The data type of the tile.
        """
        if self.kind is not None and self.kind != dtype.kind:
            raise TypeError("All tiles should have the same kind of dtype")
        if self.dtype_size is not None and self.dtype_size != dtype.itemsize:
            warnings.warn(
                f"Tile (R: {r} C: {ch} Z: {zplane}) has dtype {dtype}, which is different from one "
                f"or more of the other tiles.",
                DataFormatWarning)

        self.kind = dtype.kind
        self.dtype_size = dtype.itemsize


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
            expectations: _TileSetConsistencyDetector,
            r: int,
            ch: int,
            zplane: int,
    ) -> None:
        self._wrapped_tile = wrapped_tile
        self._expectations = expectations
        self._r = r
        self._ch = ch
        self._zplane = zplane
        self._numpy_array: np.ndarray = None

    def _load(self):
        if self._numpy_array is not None:
            return
        self._numpy_array = self._wrapped_tile.numpy_array
        self._expectations.report_dtype(self._r, self._ch, self._zplane, self._numpy_array.dtype)

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        self._load()
        raw_tile_shape = self._numpy_array.shape
        assert len(raw_tile_shape) == 2
        tile_shape = {Axes.Y: raw_tile_shape[0], Axes.X: raw_tile_shape[1]}
        self._expectations.report_tile_shape(self._r, self._ch, self._zplane, tile_shape)
        return tile_shape

    @property
    def numpy_array(self) -> np.ndarray:
        self._load()
        return self._numpy_array

    @property
    def coordinates(self) -> Mapping[Coordinates, Tuple[Number, Number]]:
        return {
            Coordinates(coordinate_name): coordinate_value
            for coordinate_name, coordinate_value in self._wrapped_tile.coordinates.items()
        }

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
        self._expectations = _TileSetConsistencyDetector()

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return self.tiles[tilekey].extras

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        return self.tiles.keys()

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
            self._expectations,
            tilekey.round, tilekey.ch, tilekey.z,
        )

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        return SlicedImageTile(
            self.tiles[TileKey(round=r, ch=ch, zplane=z)],
            self._expectations,
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
