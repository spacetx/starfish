import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties

from starfish.core.morphology.object.label_image import LabelImage
from starfish.core.morphology.object.util import _get_axes_names
from starfish.core.types import Axes, Coordinates, Number
from starfish.core.util.logging import Log
from .expand import fill_from_mask


@dataclass
class MaskData:
    binary_mask: np.ndarray
    offsets: Tuple[int, ...]
    region_properties: Optional[_RegionProperties]


class BinaryMaskCollection:
    """Collection of binary masks with a dict-like access pattern.

    Parameters
    ----------
    pixel_ticks : Union[Mapping[Axes, Sequence[int]], Mapping[str, Sequence[int]]]
        A map from the axis to the values for that axis.
    physical_ticks : Union[Mapping[Coordinates, Sequence[Number]],
                                 Mapping[str, Sequence[Number]]
        A map from the physical coordinate type to the values for axis.  For 2D label images,
        X and Y physical coordinates must be provided.  For 3D label images, Z physical
        coordinates must also be provided.
    masks : Sequence[MaskData]
        A sequence of data for binary masks.

    Attributes
    ----------
    max_shape : Mapping[Axes, int]
        Maximum index of contained masks.
    """
    def __init__(
            self,
            pixel_ticks: Union[Mapping[Axes, Sequence[int]], Mapping[str, Sequence[int]]],
            physical_ticks: Union[Mapping[Coordinates, Sequence[Number]],
                                  Mapping[str, Sequence[Number]]],
            masks: Sequence[MaskData],
            log: Optional[Log],
    ):
        self._pixel_ticks: Mapping[Axes, Sequence[int]] = {
            Axes(axis): axis_data
            for axis, axis_data in pixel_ticks.items()
        }
        self._physical_ticks: Mapping[Coordinates, Sequence[Number]] = {
            Coordinates(coord): coord_data
            for coord, coord_data in physical_ticks.items()
        }
        self._masks: MutableMapping[int, MaskData] = {}
        self._log: Log = log or Log()

        for ix, mask_data in enumerate(masks):
            if mask_data.binary_mask.ndim not in (2, 3):
                raise TypeError(f"expected 2 or 3 dimensions; got {mask_data.binary_mask.ndim}")
            if mask_data.binary_mask.dtype != np.bool:
                raise ValueError(f"expected dtype of bool; got {mask_data.binary_mask.dtype}")

            self._masks[ix] = mask_data

        if len(self._pixel_ticks) != len(self._physical_ticks):
            raise ValueError(
                "pixel_ticks should have the same cardinality as physical_ticks")
        for axis, coord in zip(*_get_axes_names(len(self._pixel_ticks))):
            if axis not in self._pixel_ticks:
                raise ValueError(f"pixel ticks missing {axis.value} data")
            if coord not in self._physical_ticks:
                raise ValueError(f"physical coordinate ticks missing {coord.value} data")
            if len(self._pixel_ticks[axis]) != len(self._physical_ticks[coord]):
                raise ValueError(
                    f"pixel ticks for {axis.name} does not have the same cardinality as physical "
                    f"coordinates ticks for {coord.name}")

    def __getitem__(self, index: int) -> xr.DataArray:
        return self._format_mask_as_xarray(index)

    def __iter__(self) -> Iterator[Tuple[int, xr.DataArray]]:
        for mask_index in self._masks.keys():
            yield mask_index, self._format_mask_as_xarray(mask_index)

    def __len__(self) -> int:
        return len(self._masks)

    def _format_mask_as_xarray(self, index: int) -> xr.DataArray:
        """Convert a np-based mask into an xarray DataArray."""
        mask_data = self._masks[index]
        max_mask_name_len = len(str(len(self._masks) - 1))

        xr_dims: MutableSequence[str] = []
        xr_coords: MutableMapping[Hashable, Any] = {}

        for ix, (axis, coord) in enumerate(zip(*_get_axes_names(len(self._pixel_ticks)))):
            xr_dims.append(axis.value)
            start_offset = mask_data.offsets[ix]
            end_offset = mask_data.offsets[ix] + mask_data.binary_mask.shape[ix]
            xr_coords[axis.value] = self._pixel_ticks[axis.value][start_offset:end_offset]
            xr_coords[coord.value] = (
                axis.value, self._physical_ticks[coord.value][start_offset:end_offset])

        return xr.DataArray(
            mask_data.binary_mask,
            dims=xr_dims,
            coords=xr_coords,
            name=f"{index:0{max_mask_name_len}d}"
        )

    def masks(self) -> Iterator[xr.DataArray]:
        for mask_index in self._masks.keys():
            yield self._format_mask_as_xarray(mask_index)

    def mask_regionprops(self, mask_id: int) -> _RegionProperties:
        """
        Return the region properties for a given mask.

        Parameters
        ----------
        mask_id : int
            The mask ID for the mask.

        Returns
        -------
        _RegionProperties
            The region properties for that mask.
        """
        mask_data = self._masks[mask_id]
        if mask_data.region_properties is None:
            # recreate the label image (but with just this mask)
            image = np.zeros(
                shape=tuple(
                    len(self._pixel_ticks[axis])
                    for axis, _ in zip(*_get_axes_names(len(self._pixel_ticks)))
                ),
                dtype=np.uint32,
            )
            fill_from_mask(
                mask_data.binary_mask,
                mask_data.offsets,
                mask_id + 1,
                image,
            )
            mask_data.region_properties = regionprops(image.data)
        return mask_data.region_properties

    @property
    def max_shape(self) -> Mapping[Axes, int]:
        return {
            axis: len(self._pixel_ticks[axis])
            for ix, (axis, _) in enumerate(zip(*_get_axes_names(len(self._pixel_ticks))))
        }

    @classmethod
    def from_label_image(cls, label_image: LabelImage) -> "BinaryMaskCollection":
        """Creates binary masks from a label image.

        Parameters
        ----------
        label_image : LabelImage
            LabelImage to extract binary masks from.

        Returns
        -------
        masks : BinaryMaskCollection
            Masks generated from the label image.
        """
        props = regionprops(label_image.xarray.data)

        pixel_ticks = {
            axis.value: label_image.xarray.coords[axis.value]
            for axis, _ in zip(*_get_axes_names(label_image.xarray.ndim))
            if axis.value in label_image.xarray.coords
        }
        physical_ticks = {
            coord.value: label_image.xarray.coords[coord.value]
            for _, coord in zip(*_get_axes_names(label_image.xarray.ndim))
            if coord.value in label_image.xarray.coords
        }
        masks: Sequence[MaskData] = [
            MaskData(prop.image, prop.bbox[:label_image.xarray.ndim], prop)
            for prop in props
        ]
        log = deepcopy(label_image.log)

        return cls(
            pixel_ticks,
            physical_ticks,
            masks,
            log,
        )

    def to_label_image(self) -> LabelImage:
        shape = tuple(
            len(self._pixel_ticks[axis])
            for axis in (Axes.ZPLANE, Axes.Y, Axes.X)
            if axis in self._pixel_ticks
        )
        label_image_array = np.zeros(shape=shape, dtype=np.uint16)
        for ix, mask_data in self._masks.items():
            fill_from_mask(
                mask_data.binary_mask,
                mask_data.offsets,
                ix + 1,
                label_image_array,
            )

        return LabelImage.from_array_and_coords(
            label_image_array,
            self._pixel_ticks,
            self._physical_ticks,
            self._log,
        )

    @classmethod
    def open_targz(cls, path: Union[str, Path]) -> "BinaryMaskCollection":
        """Load the collection saved as a .tar.gz file from disk

        Parameters
        ----------
        path : Union[str, Path]
            Path of the tar file to instantiate from.

        Returns
        -------
        masks : BinaryMaskCollection
            Collection of binary masks.
        """
        with open(os.fspath(path), "rb") as fh:
            return _io.BinaryMaskIO.read_versioned_binary_mask(fh)

    def to_targz(self, path: Union[str, Path]):
        """Save the binary masks to disk as a .tar.gz file.

        Parameters
        ----------
        path : Union[str, Path]
            Path of the tar file to write to.
        """
        with open(os.fspath(path), "wb") as fh:
            _io.BinaryMaskIO.write_versioned_binary_mask(fh, self)


# these need to be at the end to avoid recursive imports
from . import _io  # noqa
