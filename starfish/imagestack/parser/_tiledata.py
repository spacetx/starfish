from typing import Collection

from ._key import TileKey


class TileCollectionData:
    """
    Base class for a parser to implement that provides the data for a collection of tiles to be
    assembled into an ImageStack.
    """
    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        raise NotImplementedError()

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        raise NotImplementedError()

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        raise NotImplementedError()
