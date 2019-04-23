from typing import Dict, Mapping, MutableMapping, Tuple, Union

import numpy as np
import xarray as xr
from xarray.core.utils import is_scalar

from starfish.core.types import Axes, Coordinates, Number


def convert_to_selector(
        indexers: Mapping[Axes, Union[int, slice, tuple]]) -> Mapping[str, Union[int, slice]]:
    """Converts a mapping of Axis to int, slice, or tuple to a mapping of str to int or slice.  The
    latter format is required for standard xarray indexing methods.

    Parameters
    ----------
    indexers : Mapping[Axes, Union[int, slice, tuple]]
            A dictionary of dim:index where index is the value or range to index the dimension

    """
    return_dict: MutableMapping[str, Union[int, slice]] = {
        ind.value: slice(None, None) for ind in Axes}
    for key, value in indexers.items():
        if isinstance(value, tuple):
            return_dict[key.value] = slice(value[0], value[1])
        else:
            return_dict[key.value] = value
    return return_dict


def convert_coords_to_indices(array: xr.DataArray,
                              indexers: Mapping[Coordinates, Union[Number, Tuple[Number, Number]]]
                              ) -> Dict[Axes, Union[int, Tuple[Number, Number]]]:
    """
    Convert mapping of physical coordinates to value or range to mapping of corresponding Axes and
    positional coordinates.

    Parameters
    ----------
    array : xr.DataArray
        The xarray with both physical and positional coordinates.
    indexers: Mapping[Coordinates, Union[Number, Tuple[Number, Number]]]
        Mapping of physical coordinates to value or range

    Returns
    -------
    Mapping[Axes, Union[int, Tuple[Number, Number]]]:
        Mapping of Axes and positional indices that correspond to the given physical indices.

    """
    axes_indexers: Dict[Axes, Union[int, Tuple[Number, Number]]] = dict()
    if Coordinates.X in indexers:
        idx_x = find_nearest(array[Coordinates.X.value], indexers[Coordinates.X])
        axes_indexers[Axes.X] = idx_x
    if Coordinates.Y in indexers:
        idx_y = find_nearest(array[Coordinates.Y.value], indexers[Coordinates.Y])
        axes_indexers[Axes.Y] = idx_y
    if Coordinates.Z in indexers:
        idx_z = find_nearest(array[Coordinates.Z.value], indexers[Coordinates.Z])
        axes_indexers[Axes.ZPLANE] = idx_z
    return axes_indexers


def index_keep_dimensions(data: xr.DataArray,
                          indexers: Mapping[str, Union[int, slice]],
                          by_pos: bool=False
                          ) -> xr.DataArray:
    """Takes an xarray and key to index it. Indexes then adds back in lost dimensions"""
    # store original dims
    original_dims = data.dims
    # index
    if by_pos:
        data = data.isel(indexers)
    else:
        data = data.sel(indexers)
    # find missing dims
    missing_dims = set(original_dims) - set(data.dims)
    # Add back in missing dims
    data = data.expand_dims(tuple(missing_dims))

    # When the selection removes a dimension, xarray.expand_dims does not expand the non-indexed
    # dimensions that were removed.  For example, if one selects only a single zplane, it reduces
    # the z physical coordinate to a coordinate scalar, and not an array of size 1.  This hack
    # restores the dependent axes to arrays so they can be indexed.
    for primary_axis, dependent_axis in (
            (Axes.X, Coordinates.X),
            (Axes.Y, Coordinates.Y),
            (Axes.ZPLANE, Coordinates.Z),
    ):
        if primary_axis.value in missing_dims and is_scalar(data[dependent_axis.value]):
            data[dependent_axis.value] = xr.DataArray(
                np.array([data[dependent_axis.value]]),
                dims=primary_axis.value)

    # Reorder to correct format
    return data.transpose(*original_dims)


def find_nearest(array: xr.DataArray,
                 value: Union[Number, Tuple[Number, Number]]
                 ) -> Union[int, Tuple[int, int]]:
    """
    Given an xarray and value or tuple range return the indices of the closest corresponding
    value/values in the array.

    Parameters
    ----------
    array: xr.DataArray
        The array to do lookups in.

    value : Union[Number, Tuple[Number, Number]]
        The value or values to lookup.

    Returns
    -------
    Union[int, Tuple[int, int]]:
        The index or indicies of the entries closest to the given values in the array.
    """
    array = np.asarray(array)
    if isinstance(value, tuple):
        idx1 = (np.abs(array - value[0])).argmin()
        idx2 = (np.abs(array - value[1])).argmin()
        return idx1, idx2
    idx = (np.abs(array - value)).argmin()
    return idx
