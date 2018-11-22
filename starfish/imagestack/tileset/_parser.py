"""
This module parses and retains the extras metadata attached to TileSet extras.
"""

from typing import Collection, Mapping, MutableMapping

from slicedimage import TileSet

from starfish.types import Indices
from ._key import TileKey


class TileSetData:
    """
    This class parses the data, including extras, from a TileSet and its constituent tiles.
    """
    def __init__(self, tileset: TileSet) -> None:
        tile_extras: MutableMapping[TileKey, dict] = dict()
        for tile in tileset.tiles():
            round_ = tile.indices[Indices.ROUND]
            ch = tile.indices[Indices.CH]
            z = tile.indices.get(Indices.Z, 0)

            tile_extras[TileKey(round=round_, ch=ch, z=z)] = tile.extras

        self.tile_extras: Mapping[TileKey, dict] = tile_extras
        self._extras = tileset.extras

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return self.tile_extras[tilekey]

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        return self.tile_extras.keys()

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        return self._extras
