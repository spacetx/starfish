"""
This module encapsulates the logic to parse a numpy array into an ImageStack.
"""
from itertools import product
from typing import (
    Any,
    Collection,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import xarray as xr

from starfish.core.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.core.types import (
    Axes,
    Coordinates,
    Number,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes,
)


class NumpyImageTile(TileData):
    """
    This is a specialization of :py:class:`starfish.imagestack.parser.TileData` for serving data
    about a single tile coming from a numpy array.
    """
    def __init__(
            self,
            data: np.ndarray,
            coordinates: Mapping[Coordinates, Tuple[Number, Number]],
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
    def coordinates(self) -> Mapping[Coordinates, Tuple[Number, Number]]:
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
            coordinates: Optional[xr.DataArray],
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

        selectors: Mapping[str, Any] = {
            Axes.ROUND.value: r,
            Axes.CH.value: ch,
            Axes.ZPLANE.value: z,
        }

        coordinates: MutableMapping[Coordinates, Tuple[Number, Number]] = dict()

        if self.coordinates is not None:
            for coordinate_type, min_selector_value, max_selector_value in (
                    (Coordinates.X, PhysicalCoordinateTypes.X_MIN, PhysicalCoordinateTypes.X_MAX),
                    (Coordinates.Y, PhysicalCoordinateTypes.Y_MIN, PhysicalCoordinateTypes.Y_MAX),
                    (Coordinates.Z, PhysicalCoordinateTypes.Z_MIN, PhysicalCoordinateTypes.Z_MAX),
            ):
                min_selectors = dict(selectors)
                min_selectors[PHYSICAL_COORDINATE_DIMENSION] = min_selector_value.value
                max_selectors = dict(selectors)
                max_selectors[PHYSICAL_COORDINATE_DIMENSION] = max_selector_value.value

                coordinates[coordinate_type] = (
                    self.coordinates.loc[min_selectors].item(),
                    self.coordinates.loc[max_selectors].item())
        else:
            # fake coordinates!
            coordinates[Coordinates.X] = (0.0, 0.001)
            coordinates[Coordinates.Y] = (0.0, 0.001)
            coordinates[Coordinates.Z] = (0.0, 0.001)

        return NumpyImageTile(
            self.data[pos_r, pos_ch, pos_z],
            coordinates,
            {
                Axes.ROUND: r,
                Axes.CH: ch,
                Axes.ZPLANE: z,
            },
        )
