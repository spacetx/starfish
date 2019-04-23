import io
import tarfile
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from skimage.measure import regionprops

from starfish.core.types import Axes, Coordinates


AXES = [a.value for a in Axes if a not in (Axes.ROUND, Axes.CH)]
COORDS = [c.value for c in Coordinates]


def _get_axes_names(ndim: int) -> Tuple[List[str], List[str]]:
    """Get needed axes names given the number of dimensions.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    Returns
    -------
    axes : List[str]
        Axes names.
    coords : List[str]
        Coordinates names.
    """
    if ndim == 2:
        axes = [axis for axis in AXES if axis != Axes.ZPLANE.value]
        coords = [coord for coord in COORDS if coord != Coordinates.Z.value]
    elif ndim == 3:
        axes = AXES
        coords = COORDS
    else:
        raise TypeError('expected 2- or 3-D image')

    return axes, coords


def validate_segmentation_mask(arr: xr.DataArray):
    """Validate if the given array is a segmentation mask.

    Parameters
    ----------
    arr : xr.DataArray
        Array to check.
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(f"expected DataArray; got {type(arr)}")

    if arr.ndim not in (2, 3):
        raise TypeError(f"expected 2 or 3 dimensions; got {arr.ndim}")

    if arr.dtype != np.bool:
        raise TypeError(f"expected dtype of bool; got {arr.dtype}")

    axes, coords = _get_axes_names(arr.ndim)
    dims = set(axes)

    if dims != set(arr.dims):
        raise TypeError(f"missing dimensions '{dims.difference(arr.dims)}'")

    if dims.union(coords) != set(arr.coords):
        raise TypeError(f"missing coordinates '{dims.union(coords).difference(arr.coords)}'")


class SegmentationMaskCollection:
    """Collection of binary segmentation masks with a list-like access pattern.

    Parameters
    ----------
    masks : List[xr.DataArray]
        Segmentation masks.
    """
    def __init__(self, masks: List[xr.DataArray]):
        for mask in masks:
            validate_segmentation_mask(mask)

        self._masks = masks

    def __getitem__(self, index):
        return self._masks[index]

    def __iter__(self):
        return iter(self._masks)

    def __len__(self):
        return len(self._masks)

    def append(self, mask: xr.DataArray):
        """Add an existing segmentation mask.

        Parameters
        ----------
        arr : xr.DataArray
            Segmentation mask.
        """
        validate_segmentation_mask(mask)
        self._masks.append(mask)

    @classmethod
    def from_label_image(
            cls,
            label_image: np.ndarray,
            physical_ticks: Dict[Coordinates, Sequence[float]]
    ) -> "SegmentationMaskCollection":
        """Creates segmentation masks from a label image.

        Parameters
        ----------
        label_image : int array
            Integer array where each integer corresponds to a region.
        physical_ticks : Dict[Coordinates, Sequence[float]]
            Physical coordinates for each axis.

        Returns
        -------
        masks : SegmentationMaskCollection
            Masks generated from the label image.
        """
        props = regionprops(label_image)

        dims, _ = _get_axes_names(label_image.ndim)

        masks: List[xr.DataArray] = []

        coords: Dict[str, Union[list, Tuple[str, Sequence]]]

        # for each region (and its properties):
        for label, prop in enumerate(props):
            # create pixel coordinate labels from the bounding box
            # to preserve spatial indexing relative to the original image
            coords = {d: list(range(prop.bbox[i], prop.bbox[i + len(dims)]))
                      for i, d in enumerate(dims)}

            # create physical coordinate labels by taking the overlapping
            # subset from the full span of labels
            for d, c in physical_ticks.items():
                axis = d.value[0]
                i = dims.index(axis)
                coords[d.value] = (axis, c[prop.bbox[i]:prop.bbox[i + len(dims)]])

            name = str(label + 1)
            name = name.zfill(len(str(len(props))))  # pad with zeros

            mask = xr.DataArray(prop.image,
                                dims=dims,
                                coords=coords,
                                name=name)
            masks.append(mask)

        return cls(masks)

    @classmethod
    def from_disk(cls, path: str) -> "SegmentationMaskCollection":
        """Load the collection from disk.

        Parameters
        ----------
        path : str
            Path of the tar file to instantiate from.

        Returns
        -------
        masks : SegmentationMaskCollection
            Collection of segmentation masks.
        """
        masks = []

        with tarfile.open(path) as t:
            for info in t.getmembers():
                f = t.extractfile(info.name)
                mask = xr.open_dataarray(f)
                masks.append(mask)

        return cls(masks)

    def save(self, path: str):
        """Save the segmentation masks to disk.

        Parameters
        ----------
        path : str
            Path of the tar file to write to.
        """
        with tarfile.open(path, 'w:gz') as t:
            for i, mask in enumerate(self._masks):
                data = mask.to_netcdf()
                with io.BytesIO(data) as buff:
                    info = tarfile.TarInfo(name=str(i) + '.nc')
                    info.size = len(data)
                    t.addfile(tarinfo=info, fileobj=buff)
