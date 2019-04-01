import itertools
import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from skimage.measure import regionprops

from starfish.types import Axes, Coordinates


AXES = [a.value for a in Axes if a not in (Axes.ROUND, Axes.CH)]
COORDS = [c.value for c in Coordinates]


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

    if arr.ndim == 2:
        axes = AXES[1:]
        coords = COORDS[1:]
    else:
        axes = AXES
        coords = COORDS

    for dim in axes:
        if dim not in arr.dims:
            raise TypeError(f"no dimension '{dim}'")

    for coord in itertools.chain(axes, coords):
        if coord not in arr.coords:
            raise TypeError(f"no coordinate '{coord}'")


class SegmentationMaskCollection:
    """Collection of binary segmentation masks with a list-like access pattern.

    Parameters
    ----------
    masks : list of xr.DataArray
        Segmentation masks.
    """
    _masks: List[xr.DataArray]

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

    def add_mask(self, mask: xr.DataArray):
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
            physical_ticks: Dict[Coordinates, List[float]]
    ) -> "SegmentationMaskCollection":
        """Creates segmentation masks from a label image.

        Parameters
        ----------
        label_image : int array
            Integer array where each integer corresponds to a cell.
        physical_ticks : Dict[Coordinates, List[float]]
            Physical coordinates for each axis.

        Returns
        -------
        masks : SegmentationMaskCollection
            Masks generated from the label image.
        """
        props = regionprops(label_image)

        if label_image.ndim == 2:
            dims = AXES[1:]
        elif label_image.ndim == 3:
            dims = AXES
        else:
            raise TypeError('expected 2- or 3-D image')

        masks: List[xr.DataArray] = []

        coords: Dict[str, Union[list, Tuple[str, list]]]

        for label, prop in enumerate(props):
            coords = {d: list(range(prop.bbox[i], prop.bbox[i + len(dims)]))
                      for i, d in enumerate(dims)}

            for d, c in physical_ticks.items():
                axis = d.value[0]
                i = dims.index(axis)
                coords[d.value] = (axis, c[prop.bbox[i]:prop.bbox[i + len(dims)]])

            mask = xr.DataArray(prop.image,
                                dims=dims,
                                coords=coords,
                                name=str(label + 1))
            masks.append(mask)

        return cls(masks)

    @classmethod
    def from_disk(cls, path: str) -> "SegmentationMaskCollection":
        """Load the collection from disk.

        Parameters
        ----------
        path : str
            Path of the directory to instantiate from.

        Returns
        -------
        masks : SegmentationMaskCollection
            Collection of segmentation masks.
        """
        masks = []
        for p in os.listdir(path):
            mask = xr.open_dataarray(osp.join(path, p))
            masks.append(mask)

        return cls(masks)

    def save(self, path: str, overwrite: bool = False):
        """Save the segmentation masks to disk.

        Parameters
        ----------
        path : str
            Path of the directory to write to.
        overwrite : bool, optional
            Whether to overwrite the directory if it exists.
        """
        try:
            os.mkdir(path)
        except FileExistsError:
            if not overwrite:
                raise
            shutil.rmtree(path, ignore_errors=True)
            os.mkdir(path)

        for i, mask in enumerate(self._masks):
            mask.to_netcdf(osp.join(path, str(i)), 'w')
