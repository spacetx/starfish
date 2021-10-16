"""
This module encapsulates the logic to parse a numpy array into an ImageStack.
"""
from itertools import product
from typing import (
    cast,
    Collection,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
)

import numpy as np

from starfish.core.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.core.types import (
    ArrayLike,
    Axes,
    Coordinates,
    Number,
)


class NumpyImageTile(TileData):
    """
    This is a specialization of :py:class:`starfish.imagestack.parser.TileData` for serving data
    about a single tile coming from a numpy array.
    """
    def __init__(
            self,
            data: np.ndarray,
            coordinates: Mapping[Coordinates, ArrayLike[Number]],
            selector: Mapping[Axes, int],
    ) -> None:
        self._data = data
        self._coordinates = coordinates
        self._selector = selector

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        raw_tile_shape = self._data.shape
        assert len(raw_tile_shape) == 2
        tile_shape = {Axes.Y: raw_tile_shape[0], Axes.X: raw_tile_shape[1]}
        return tile_shape

    @property
    def numpy_array(self):
        return self._data

    @property
    def coordinates(self) -> Mapping[Coordinates, ArrayLike[Number]]:
        return self._coordinates

    @property
    def selector(self) -> Mapping[Axes, int]:
        return self._selector


class NumpyData(TileCollectionData):
    """
    This is a specialization of :py:class:`starfish.imagestack.parser.TileCollectionData` for
    serving tile data from a numpy array.
    """
    def __init__(
            self,
            data: np.ndarray,
            index_labels: Mapping[Axes, Sequence[int]],
            coordinates: Optional[Mapping[Coordinates, ArrayLike[Number]]],
    ) -> None:
        self.data = data
        self.index_labels = index_labels
        self.coordinates = coordinates

    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        return {}

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        keys: MutableSequence[TileKey] = list()
        axis_names: MutableSequence[Axes] = list()
        labels: MutableSequence[Sequence[int]] = list()
        for index_name, index_labels in self.index_labels.items():
            axis_names.append(index_name)
            labels.append(index_labels)

        for indices in product(*labels):
            selector: MutableMapping[Axes, int] = dict()
            for index_name, index_value in zip(axis_names, indices):
                selector[index_name] = index_value

            keys.append(
                TileKey(
                    round=selector[Axes.ROUND],
                    ch=selector[Axes.CH],
                    zplane=selector[Axes.ZPLANE],
                )
            )

        return keys

    @property
    def group_by(self) -> Set[Axes]:
        """Returns the axes to group by when we load the data."""
        return set()

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self.data.shape[-2], Axes.X: self.data.shape[-1]}

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        return {}

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        return self.get_tile(tilekey.round, tilekey.ch, tilekey.z)

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        # find the positional r/ch/z.
        pos_r = self.index_labels[Axes.ROUND].index(r)
        pos_ch = self.index_labels[Axes.CH].index(ch)
        pos_z = self.index_labels[Axes.ZPLANE].index(z)

        coordinates: MutableMapping[Coordinates, ArrayLike[Number]] = dict()

        if self.coordinates is not None:
            coordinates[Coordinates.X] = self.coordinates[Coordinates.X]
            coordinates[Coordinates.Y] = self.coordinates[Coordinates.Y]
            if Coordinates.Z in self.coordinates:
                z_coord = cast(Number, self.coordinates[Coordinates.Z][pos_z])
                coordinates[Coordinates.Z] = [z_coord]
        else:
            # fake coordinates!
            coordinates[Coordinates.X] = np.linspace(0.0, 0.001, self.data.shape[-1])
            coordinates[Coordinates.Y] = np.linspace(0.0, 0.001, self.data.shape[-2])
            coordinates[Coordinates.Z] = [np.linspace(0.0, 0.001, self.data.shape[-3])[pos_z]]

        return NumpyImageTile(
            self.data[pos_r, pos_ch, pos_z],
            coordinates,
            {
                Axes.ROUND: r,
                Axes.CH: ch,
                Axes.ZPLANE: z,
            },
        )
