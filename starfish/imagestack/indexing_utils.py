from typing import Mapping, Union

import xarray as xr

from starfish.types import Indices


def convert_to_indexers_dict(indexers) -> Mapping[str, Union[int, slice]]:
    """Converts a dict of indexers from Dict[Indices, (int/tuple)] to
    Dict[str, (int/slice)] so that it can be passed into standard xarray
    indexing methods

    Parameters
    ----------
    indexers : Dict[Indices, (int/tuple)]
            A dictionary of dim:index where index is the value or range to index the dimension

    """
    return_dict = {ind.value: slice(None, None) for ind in Indices}
    for key, value in indexers.items():
        if isinstance(value, tuple):
            value = slice(value[0], value[1])
        return_dict[key] = value
    return return_dict


def index_keep_dimensions(data: xr.DataArray, indexers: Mapping[str, Union[int, slice]]
                          ) -> xr.DataArray:
    """Takes an xarray and key to index it. Indexes then adds back in lost dimensions"""
    # store original dims
    original_dims = data.dims
    # index
    data = data.sel(indexers)
    # find missing dims
    missing_dims = set(original_dims) - set(data.dims)
    # Add back in missing dims
    data = data.expand_dims(tuple(missing_dims))
    # Reorder to correct format
    return data.transpose(*original_dims)
