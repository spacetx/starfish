from collections import OrderedDict
from typing import Collection, List, Mapping, MutableSequence, Optional, Tuple, Union

import numpy as np
from slicedimage import TileSet

from starfish.core.imagestack.parser import TileCollectionData, TileData, TileKey
from starfish.core.imagestack.physical_coordinate_calculator import (
    recalculate_physical_coordinate_range,
)
from starfish.core.types import Axes, Coordinates, Number


class CropParameters:
    """Parameters for cropping an ImageStack at load time."""
    def __init__(
            self,
            *,
            permitted_rounds: Optional[Collection[int]]=None,
            permitted_chs: Optional[Collection[int]]=None,
            permitted_zplanes: Optional[Collection[int]]=None,
            x_slice: Optional[Union[int, slice]]=None,
            y_slice: Optional[Union[int, slice]]=None,
    ):
        """
        Parameters
        ----------
        permitted_rounds : Optional[Collection[int]]
            The rounds in the original dataset to load into the ImageStack.  If this is not set,
            then all rounds are loaded into the ImageStack.
        permitted_chs : Optional[Collection[int]]
            The channels in the original dataset to load into the ImageStack.  If this is not set,
            then all channels are loaded into the ImageStack.
        permitted_zplanes : Optional[Collection[int]]
            The z-layers in the original dataset to load into the ImageStack.  If this is not set,
            then all z-layers are loaded into the ImageStack.
        x_slice : Optional[Union[int, slice]]
            The x-range in the x-y tile that is loaded into the ImageStack.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.
        y_slice : Optional[Union[int, slice]]
            The y-range in the x-y tile that is loaded into the ImageStack.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.
        """
        self._permitted_rounds = set(permitted_rounds) if permitted_rounds else None
        self._permitted_chs = set(permitted_chs) if permitted_chs else None
        self._permitted_zplanes = set(permitted_zplanes) if permitted_zplanes else None
        self._x_slice = x_slice
        self._y_slice = y_slice

    def _add_permitted_axes(self, axis_type: Axes, permitted_axis: int) -> None:
        """
        Add a value to one of the permitted axes sets.
        """
        if axis_type == Axes.ROUND and self._permitted_rounds:
            self._permitted_rounds.add(permitted_axis)
        if axis_type == Axes.CH and self._permitted_chs:
            self._permitted_chs.add(permitted_axis)
        if axis_type == Axes.ZPLANE and self._permitted_zplanes:
            self._permitted_zplanes.add(permitted_axis)

    def further_crop(self, further_crop_params) -> "CropParameters":
        if (further_crop_params._permitted_chs
                and not further_crop_params._permitted_chs.issubset(self._permitted_chs)):
            raise ValueError("Can only provide further ch crop params within the existing set of"
                             "ch crop params")
        else:
            self._permitted_chs = further_crop_params._permitted_chs

        if (further_crop_params._permitted_rounds
                and not further_crop_params._permitted_rounds.issubset(self._permitted_rounds)):
            raise ValueError("Can only provide further round crop params within the existing set of"
                             "round crop params")
        else:
            self._permitted_rounds = further_crop_params._permitted_rounds

        if (further_crop_params._permitted_zplanes
                and not further_crop_params._permitted_zplanes.issubset(self._permitted_zplanes)):
            raise ValueError("Can only provide further zplane crop params within the existing set "
                             "of zplane crop params")
        else:
            self._permitted_zplanes = further_crop_params._permitted_zplanes
        self._x_slice = further_crop_params._x_slice
        self._y_slice = further_crop_params._y_slice
        return self

    def filter_tilekeys(self, tilekeys: Collection[TileKey]) -> Collection[TileKey]:
        """
        Filters tilekeys for those that should be included in the resulting ImageStack.
        """
        results: MutableSequence[TileKey] = list()
        for tilekey in tilekeys:
            if self._permitted_rounds is not None and tilekey.round not in self._permitted_rounds:
                continue
            if self._permitted_chs is not None and tilekey.ch not in self._permitted_chs:
                continue
            if self._permitted_zplanes is not None and tilekey.z not in self._permitted_zplanes:
                continue

            results.append(tilekey)

        return results

    @staticmethod
    def _crop_axis(size: int, crop: Optional[Union[int, slice]]) -> Tuple[int, int]:
        """
        Given the size of along an axis, and an optional cropping, return the start index
        (inclusive) and end index (exclusive) of the crop.  If no crop is specified, then the
        original size (0, size) is returned.
        """
        # convert int crops to a slice operation.
        if isinstance(crop, int):
            if crop < 0 or crop >= size:
                raise IndexError("crop index out of range")
            return crop, crop + 1

        # convert start and stop to absolute values.
        start: int
        if crop is None or crop.start is None:
            start = 0
        elif crop.start is not None and crop.start < 0:
            start = max(0, size + crop.start)
        else:
            start = min(size, crop.start)

        stop: int
        if crop is None or crop.stop is None:
            stop = size
        elif crop.stop is not None and crop.stop < 0:
            stop = max(0, size + crop.stop)
        else:
            stop = min(size, crop.stop)

        return start, stop

    @staticmethod
    def parse_coordinate_groups(tileset: TileSet) -> List["CropParameters"]:
        """Takes a tileset and compares the physical coordinates on each tile to
         create aligned coordinate groups (groups of tiles that have the same physical coordinates)

         Returns
         -------
         A list of CropParameters. Each entry describes the r/ch/z values of tiles that are aligned
         (have matching coordinates)
         """
        coord_groups: OrderedDict[tuple, CropParameters] = OrderedDict()
        for tile in tileset.tiles():
            x_y_coords = (
                tile.coordinates[Coordinates.X][0], tile.coordinates[Coordinates.X][1],
                tile.coordinates[Coordinates.Y][0], tile.coordinates[Coordinates.Y][1]
            )
            # A tile with this (x, y) has already been seen, add tile's Indices to CropParameters
            if x_y_coords in coord_groups:
                crop_params = coord_groups[x_y_coords]
                crop_params._add_permitted_axes(Axes.CH, tile.indices[Axes.CH])
                crop_params._add_permitted_axes(Axes.ROUND, tile.indices[Axes.ROUND])
                if Axes.ZPLANE in tile.indices:
                    crop_params._add_permitted_axes(Axes.ZPLANE, tile.indices[Axes.ZPLANE])
            else:
                coord_groups[x_y_coords] = CropParameters(
                    permitted_chs=[tile.indices[Axes.CH]],
                    permitted_rounds=[tile.indices[Axes.ROUND]],
                    permitted_zplanes=[tile.indices[Axes.ZPLANE]] if Axes.ZPLANE in tile.indices
                    else None)
        return list(coord_groups.values())

    def crop_shape(self, shape: Mapping[Axes, int]) -> Mapping[Axes, int]:
        """
        Given the shape of the original tile, return the shape of the cropped tile.
        """
        output_x_shape = CropParameters._crop_axis(shape[Axes.X], self._x_slice)
        output_y_shape = CropParameters._crop_axis(shape[Axes.Y], self._y_slice)
        width = output_x_shape[1] - output_x_shape[0]
        height = output_y_shape[1] - output_y_shape[0]

        return {Axes.Y: height, Axes.X: width}

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Given the original image, return the cropped image.
        """
        output_x_shape = CropParameters._crop_axis(image.shape[1], self._x_slice)
        output_y_shape = CropParameters._crop_axis(image.shape[0], self._y_slice)

        return image[output_y_shape[0]:output_y_shape[1], output_x_shape[0]:output_x_shape[1]]

    def crop_coordinates(
            self,
            coordinates: Mapping[Coordinates, Tuple[Number, Number]],
            shape: Mapping[Axes, int],
    ) -> Mapping[Coordinates, Tuple[Number, Number]]:
        """
        Given a mapping of coordinate to coordinate values, return a mapping of the coordinate to
        cropped coordinate values.
        """
        xmin, xmax = coordinates[Coordinates.X]
        ymin, ymax = coordinates[Coordinates.Y]
        if self._x_slice is not None:
            xmin, xmax = recalculate_physical_coordinate_range(
                xmin, xmax,
                shape[Axes.X],
                self._x_slice)
        if self._y_slice is not None:
            ymin, ymax = recalculate_physical_coordinate_range(
                ymin, ymax,
                shape[Axes.Y],
                self._y_slice)

        return_coords = {
            Coordinates.X: (xmin, xmax),
            Coordinates.Y: (ymin, ymax)
        }
        if Coordinates.Z in coordinates:
            return_coords[Coordinates.Z] = coordinates[Coordinates.Z]
        return return_coords


class CroppedTileData(TileData):
    """Represent a cropped view of a TileData object."""
    def __init__(self, tile_data: TileData, cropping_parameters: CropParameters):
        self.backing_tile_data = tile_data
        self.cropping_parameters = cropping_parameters

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return self.cropping_parameters.crop_shape(self.backing_tile_data.tile_shape)

    @property
    def numpy_array(self) -> np.ndarray:
        return self.cropping_parameters.crop_image(self.backing_tile_data.numpy_array)

    @property
    def coordinates(self) -> Mapping[Coordinates, Tuple[Number, Number]]:
        return self.cropping_parameters.crop_coordinates(
            self.backing_tile_data.coordinates,
            self.backing_tile_data.tile_shape,
        )

    @property
    def selector(self) -> Mapping[Axes, int]:
        return self.backing_tile_data.selector


class CroppedTileCollectionData(TileCollectionData):
    """Represent a cropped view of a TileCollectionData object."""
    def __init__(
            self,
            backing_tile_collection_data: TileCollectionData,
            crop_parameters: CropParameters,
    ) -> None:
        self.backing_tile_collection_data = backing_tile_collection_data
        self.crop_parameters = crop_parameters

    def __getitem__(self, tilekey: TileKey) -> dict:
        return self.backing_tile_collection_data[tilekey]

    def keys(self) -> Collection[TileKey]:
        return self.crop_parameters.filter_tilekeys(self.backing_tile_collection_data.keys())

    @property
    def tile_shape(self) -> Mapping[Axes, int]:
        return self.crop_parameters.crop_shape(self.backing_tile_collection_data.tile_shape)

    @property
    def extras(self) -> dict:
        return self.backing_tile_collection_data.extras

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        return CroppedTileData(
            self.backing_tile_collection_data.get_tile_by_key(tilekey),
            self.crop_parameters,
        )

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        return CroppedTileData(
            self.backing_tile_collection_data.get_tile(r, ch, z),
            self.crop_parameters,
        )
