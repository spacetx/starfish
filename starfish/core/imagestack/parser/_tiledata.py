from typing import Collection, Mapping, Set

import numpy as np

from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from ._key import TileKey


class TileData:
    """
    Base class for a parser to implement that provides the data for a single tile.
    """
    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        raise NotImplementedError()

    @property
    def numpy_array(self) -> np.ndarray:
        """Return the image data representing the tile.  The tile must be row-major.

        Returns
        -------
        ndarray :
            The image data
        """
        raise NotImplementedError()

    @property
    def coordinates(self) -> Mapping[Coordinates, ArrayLike[Number]]:
        raise NotImplementedError()

    @property
    def selector(self) -> Mapping[Axes, int]:
        raise NotImplementedError()


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
    def tile_shape(self) -> Mapping[Axes, int]:
        """Returns the shape of a tile."""
        raise NotImplementedError()

    @property
    def group_by(self) -> Set[Axes]:
        """Returns the axes to group by when we load the data."""
        raise NotImplementedError()

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        raise NotImplementedError()

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        raise NotImplementedError()

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        raise NotImplementedError()
