import io
import itertools
import tarfile
from dataclasses import dataclass
from typing import (
    cast,
    Dict,
    Hashable,
    Iterable,
    Iterator,
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

from starfish.core.types import Axes, Coordinates
from .expand import fill_from_mask
from .util import _get_axes_names, AXES_ORDER


def _validate_segmentation_mask(arr: xr.DataArray):
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


@dataclass
class MaskData:
    binary_mask: xr.DataArray
    region_properties: Optional[_RegionProperties]


class SegmentationMaskCollection:
    """Collection of binary segmentation masks with a dict-like access pattern.

    Parameters
    ----------
    masks : Iterable[xr.DataArray]
        Segmentation masks.
    props : Iterable[_RegionProperties]
        Properties for each of the regions in the masks.

    Attributes
    ----------
    max_shape : Dict[Axes, Optional[int]]
        Maximum index of contained masks.
    """
    def __init__(
            self,
            masks: Iterable[xr.DataArray],
            props: Optional[Iterable[Optional[_RegionProperties]]] = None,
    ):
        if props is None:
            props = itertools.cycle((None,))
        self._masks: MutableMapping[int, MaskData] = {}
        self.max_shape: Dict[Axes, int] = {
            Axes.X: 0,
            Axes.Y: 0,
            Axes.ZPLANE: 0
        }

        for ix, (mask, mask_props) in enumerate(zip(masks, props)):
            _validate_segmentation_mask(mask)

            self._masks[ix] = MaskData(mask, mask_props)

            for axis in Axes:
                if axis.value in mask.coords:
                    max_val = mask.coords[axis.value].values[-1]
                    if max_val >= self.max_shape[axis]:
                        self.max_shape[axis] = max_val + 1

    def __getitem__(self, index: int) -> xr.DataArray:
        return self._masks[index].binary_mask

    def __iter__(self) -> Iterator[Tuple[int, xr.DataArray]]:
        for mask_index, mask_data in self._masks.items():
            yield mask_index, mask_data.binary_mask

    def __len__(self) -> int:
        return len(self._masks)

    def masks(self) -> Iterator[xr.DataArray]:
        for mask_index, mask_data in self._masks.items():
            yield mask_data.binary_mask

    def mask_regionprops(self, mask_id: int) -> _RegionProperties:
        """
        Return the region properties for
        Parameters
        ----------
        mask_id

        Returns
        -------

        """
        mask_data = self._masks[mask_id]
        if mask_data.region_properties is None:
            # recreate the label image (but with just this mask)
            image = np.zeros(
                shape=tuple(
                    self.max_shape[axis]
                    for axis in AXES_ORDER
                    if self.max_shape[axis] != 0
                ),
                dtype=np.uint32,
            )
            fill_from_mask(
                mask_data.binary_mask,
                mask_id + 1,
                image,
                [axis for axis in AXES_ORDER if self.max_shape[axis] > 0],
            )
            mask_data.region_properties = regionprops(image)
        return mask_data.region_properties

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

        masks: MutableSequence[xr.DataArray] = []
        coords: Dict[Hashable, Union[list, Tuple[str, Sequence]]]

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

        return cls(masks, props)

    def to_label_image(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            *,
            ordering: Sequence[Axes] = AXES_ORDER,
    ):
        """Create a label image from the contained masks.

        Parameters
        ----------
        shape : Optional[Tuple[int, ...]]
            Shape of the label image. If ``None``, use maximum index for each axis.
        ordering : Sequence[Axes]
            Ordering of the axes. Default is z, y, x.

        Returns
        -------
        label_image : np.ndarray
            uint16 array where each integer corresponds to a label
        """
        ordering = [o for o in ordering if self.max_shape[o] > 0]

        max_shape = tuple(self.max_shape[o] for o in ordering)

        if shape is None:
            shape = max_shape
        elif np.any(np.less(shape, max_shape)):
            raise ValueError("shape less than the maximum of the data provided."
                             "cropping is not supported at this time")

        label_image = np.zeros(shape, dtype=np.uint16)

        for i, mask in iter(self):
            fill_from_mask(mask, i + 1, label_image, ordering)

        return label_image

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
            for i, mask in iter(self):
                data = cast(bytes, mask.to_netcdf())
                with io.BytesIO(data) as buff:
                    info = tarfile.TarInfo(name=str(i) + '.nc')
                    info.size = len(data)
                    t.addfile(tarinfo=info, fileobj=buff)
