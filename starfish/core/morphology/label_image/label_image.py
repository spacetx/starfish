from pathlib import Path
from typing import Any, Hashable, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import xarray as xr
from semantic_version import Version

from starfish.core.morphology.util import (
    _get_axes_names,
    _normalize_physical_ticks,
    _normalize_pixel_ticks,
)
from starfish.core.types import ArrayLike, Axes, Coordinates, LOG, Number, STARFISH_EXTRAS_KEY
from starfish.core.util.logging import Log


class AttrKeys:
    DOCTYPE = f"{STARFISH_EXTRAS_KEY}.DOCTYPE"
    LOG = f"{STARFISH_EXTRAS_KEY}.{LOG}"
    VERSION = f"{STARFISH_EXTRAS_KEY}.VERSION"


DOCTYPE_STRING = "starfish/LabelImage"
CURRENT_VERSION = Version("0.0.0")
MIN_SUPPORTED_VERSION = Version("0.0.0")
MAX_SUPPORTED_VERSION = Version("0.0.0")


class LabelImage:
    """Wraps an xarray that contains a 2D or 3D labeled image.  Each axis is labeled with physical
    coordinate data."""

    def __init__(self, label_image: xr.DataArray):
        # verify that the data array has the required elements.
        if label_image.dtype.kind not in ("i", "u"):
            raise TypeError("label image should be an integer type")
        for axis in (Axes.X, Axes.Y):
            if axis.value not in label_image.coords:
                raise ValueError(f"label image should have an {axis.value} axis")
        expected_physical_coordinates: Tuple[Coordinates, ...]
        if label_image.ndim == 5:
            expected_physical_coordinates = (Coordinates.X, Coordinates.Y, Coordinates.Z)
        else:
            expected_physical_coordinates = (Coordinates.X, Coordinates.Y)
        for coord in expected_physical_coordinates:
            if coord.value not in label_image.coords:
                raise ValueError(f"label image should have a {coord.value} coordinates")

        self.label_image = label_image.copy(deep=False)
        if AttrKeys.DOCTYPE not in self.label_image.attrs:
            self.label_image.attrs[AttrKeys.DOCTYPE] = DOCTYPE_STRING
        if AttrKeys.LOG not in self.label_image.attrs:
            self.label_image.attrs[AttrKeys.LOG] = Log().encode()

    @classmethod
    def from_label_array_and_ticks(
            cls,
            array: np.ndarray,
            pixel_ticks: Optional[Union[
                Mapping[Axes, ArrayLike[int]],
                Mapping[str, ArrayLike[int]]]],
            physical_ticks: Union[
                Mapping[Coordinates, ArrayLike[Number]],
                Mapping[str, ArrayLike[Number]]],
            log: Optional[Log],
    ) -> "LabelImage":
        """Constructs a LabelImage from an array containing the labels, a set of physical
        coordinates, and an optional log of how this label image came to be.

        Parameters
        ----------
        array : np.ndarray
            A 2D or 3D array containing the labels.  The ordering of the axes must be Y, X for 2D
            images and ZPLANE, Y, X for 3D images.
        pixel_ticks : Optional[Union[Mapping[Axes, ArrayLike[int]], Mapping[str, ArrayLike[int]]]]
            A map from the axis to the values for that axis.  For any axis that exist in the array
            but not in pixel_coordinates, the pixel coordinates are assigned from 0..N-1, where N is
            the size along that axis.
        physical_ticks : Union[Mapping[Coordinates, ArrayLike[Number]], Mapping[str,
        ArrayLike[Number]]]
            A map from the physical coordinate type to the values for axis.  For 2D label images,
            X and Y physical coordinates must be provided.  For 3D label images, Z physical
            coordinates must also be provided.
        log : Optional[Log]
            A log of how this label image came to be.
        """
        # normalize the pixel coordinates to Mapping[Axes, ArrayLike[int]]
        pixel_ticks = _normalize_pixel_ticks(pixel_ticks)
        # normalize the physical coordinates to Mapping[Coordinates, ArrayLike[Number]]
        physical_ticks = _normalize_physical_ticks(physical_ticks)

        img_axes, img_coords = _get_axes_names(array.ndim)
        xr_axes = [axis.value for axis in img_axes]
        try:
            xr_coords: MutableMapping[Hashable, Any] = {
                coord.value: (axis.value, physical_ticks[coord])
                for axis, coord in zip(img_axes, img_coords)
            }
        except KeyError as ex:
            raise KeyError(f"missing physical coordinates {ex.args[0]}") from ex

        for ix, axis in enumerate(img_axes):
            xr_coords[axis.value] = pixel_ticks.get(axis, np.arange(0, array.shape[ix]))

        dataarray = xr.DataArray(
            array,
            dims=xr_axes,
            coords=xr_coords,
        )
        dataarray.attrs.update({
            AttrKeys.LOG: (log or Log()).encode(),
            AttrKeys.DOCTYPE: DOCTYPE_STRING,
            AttrKeys.VERSION: str(CURRENT_VERSION),
        })

        return LabelImage(dataarray)

    @property
    def xarray(self):
        """Returns the xarray that contains the label image and the physical coordinates."""
        return self.label_image

    @property
    def log(self) -> Log:
        """Returns a copy of the provenance data.  Modifications to this copy will not affect the
        log stored on this label image."""
        return Log.decode(self.label_image.attrs[AttrKeys.LOG])

    @classmethod
    def open_netcdf(cls, path: Union[str, Path]) -> "LabelImage":
        """Load a label image saved as a netcdf file from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path of the label image to instantiate from.

        Returns
        -------
        label_image : LabelImage
            Label image from the path.
        """
        label_image = xr.open_dataarray(path)
        if (
                AttrKeys.DOCTYPE not in label_image.attrs
                or label_image.attrs[AttrKeys.DOCTYPE] != DOCTYPE_STRING
                or AttrKeys.VERSION not in label_image.attrs
        ):
            raise ValueError(f"{path} does not appear to be a starfish label image")
        if not (
                MIN_SUPPORTED_VERSION
                <= Version(label_image.attrs[AttrKeys.VERSION])
                <= MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"{path} contains a label image, but the version "
                f"{label_image.attrs[AttrKeys.VERSION]} is not supported")

        return cls(label_image)

    def to_netcdf(self, path: Union[str, Path]):
        """Save the label image as a netcdf file.

        Parameters
        ----------
        path : Union[str, Path]
            Path of the netcdf file to write to.
        """
        self.label_image.to_netcdf(path)
