from typing import Union

import numpy as np
import xarray as xr

def preserve_float_range(
        array: Union[xr.DataArray, np.ndarray],
        rescale: bool=False) -> Union[xr.DataArray, np.ndarray]:
    """
    Clips values below zero to zero. If values above one are detected, clips them
    to 1 unless `rescale` is True, in which case the input is scaled by
    the max value and the dynamic range is preserved.

    Parameters
    ----------
    array : Union[xr.DataArray, np.ndarray]
        Array whose values should be in the interval [0, 1] but may not be.
    rescale: bool
        If true, scale values by the max.

    Returns
    -------
    array : Union[xr.DataArray, np.ndarray]
        Array whose values are in the interval [0, 1].

    """
    array = array.copy()

    if isinstance(array, xr.DataArray):
        data = array.values
    else:
        data = array

    negative = data < 0
    if np.any(negative):
        data[negative] = 0
    if np.any(data > 1):
        if rescale:
            data /= data.max()
        else:
            data[data > 1] = 1
    return array.astype(np.float32)
