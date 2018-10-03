"""
This module describes the contracts to provide data to the experiment builder.
"""

from typing import Mapping, Tuple, Union

import numpy as np
from slicedimage import (
    ImageFormat,
)

from starfish.types import Coordinates, Number


class FetchedTile:
    """
    This is the contract for providing the data for constructing a :class:`slicedimage.Tile`.
    """
    def __init__(self, *args, **kwargs):
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return Tile shape.

        Returns
        -------
        Tuple[int, ...]
            The tile shape in (y, x)
        """
        raise NotImplementedError()

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        """Return the tile's coordinates in the global coordinate space..

        Returns
        -------
        Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]
            Maps from a coordinate type (e.g. 'x', 'y', or 'z') to its value or range.
        """
        raise NotImplementedError()

    @property
    def extras(self) -> dict:
        """Return the extras data associated with the tile.

        Returns
        -------
        Mapping[str, Any]
            Maps from a key to its value.
        """
        return {}

    @property
    def format(self) -> ImageFormat:
        """Return the Tile's format

        Returns
        -------
        ImageFormat :
            a slicedimage format type, e.g. ImageFormat.TIFF
        """
        raise NotImplementedError()

    def tile_data(self) -> np.ndarray:
        """Return the image data representing the tile.

        Returns
        -------
        ndarray :
            The image data
        """
        raise NotImplementedError()


class TileFetcher:
    """
    This is the contract for providing the image data for constructing a
    :class:`slicedimage.Collection`.
    """
    def get_tile(self, fov: int, hyb: int, ch: int, z: int) -> FetchedTile:
        """
        Given fov, hyb, ch, and z, return an instance of a :class:`.FetchedImage` that can be
        queried for the image data.
        """
        raise NotImplementedError()
