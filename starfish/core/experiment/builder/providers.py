"""
This module describes the contracts to provide data to the experiment builder.
"""

from typing import Mapping, Union

import numpy as np

from starfish.core.types import Axes, Coordinates, CoordinateValue


class FetchedTile:
    """
    This is the contract for providing the data for constructing a :class:`slicedimage.Tile`.
    """
    def __init__(self, *args, **kwargs):
        pass

    @property
    def shape(self) -> Mapping[Axes, int]:
        """Return Tile shape.

        Returns
        -------
        Mapping[Axis, int]
            The shape of the tile, mapping from Axes to its size.
        """
        raise NotImplementedError()

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Return the tile's coordinates in the global coordinate space..

        Returns
        -------
        Mapping[Union[str, Coordinates], CoordinateValue]
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

    def tile_data(self) -> np.ndarray:
        """Return the image data representing the tile.  The tile must be row-major.

        Returns
        -------
        ndarray :
            The image data
        """
        raise NotImplementedError()


class TileFetcher:
    """
    This is the contract for providing the image data for constructing a
    :py:class:`slicedimage.Collection`.
    """
    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        """
        Given fov_id, round_label, ch_label, and zplane_label, return an instance of a
        :py:class:`.FetchedTile` that can be queried for the image data.
        """
        raise NotImplementedError()
