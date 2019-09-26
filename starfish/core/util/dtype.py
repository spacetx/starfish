from typing import TypeVar

import numpy as np
import xarray as xr


PreserveFloatRangeType = TypeVar("PreserveFloatRangeType", xr.DataArray, np.ndarray)


def preserve_float_range(
        array: PreserveFloatRangeType,
        rescale: bool = False,
        preserve_input: bool = False,
) -> PreserveFloatRangeType:
    """
    Clips values below zero to zero. If values above one are detected, clips them
    to 1 unless `rescale` is True, in which case the input is scaled by
    the max value and the dynamic range is preserved.

    Input arrays may be modified by this operation if `preserve_input` is False.  There is no
    guarantee that the input array is returned even if `preserve_input` is True, however.

    Parameters
    ----------
    array : Union[xr.DataArray, np.ndarray]
        Array whose values should be in the interval [0, 1] but may not be.
    rescale : bool
        If true, scale values by the max.
    preserve_input : bool
        If True, ensure that we do not modify the input data.  This may either be done by making a
        copy of the input data or ensuring that the operation does not modify the input array.

        Even if `preserve_input` is True, modifications to the resulting array may modify the input
        array.

    Returns
    -------
    array : Union[xr.DataArray, np.ndarray]
        Array whose values are in the interval [0, 1].

    """
    if isinstance(array, xr.DataArray):
        # do a shallow copy
        array = array.copy(deep=False)
        data = array.values
    else:
        data = array

    casted = data.astype(np.float32, copy=False)
    if casted is not data:
        preserve_input = False
    data = casted

    if preserve_input or not data.flags['WRITEABLE']:
        # if we still want a copy, check to see if any modifications would be made.  if so, make a
        # copy.
        belowzero = np.any(data < 0)
        aboveone = np.any(data > 1)
        if belowzero or aboveone:
            data = data.copy()
            _preserve_float_range_in_place(data, rescale and aboveone)
    else:
        # we don't want a copy, so we just do it in place.
        aboveone = np.any(data > 1)
        _preserve_float_range_in_place(data, rescale and aboveone)

    if isinstance(array, xr.DataArray):
        if data is not array.values:
            array.values = data
    else:
        array = data

    return array


def _preserve_float_range_in_place(
        array: np.ndarray,
        rescale: bool = False,
):
    """
    Clips values below zero to zero. If values above one are detected, clips them
    to 1 unless `rescale` is True, in which case the input is scaled by
    the max value and the dynamic range is preserved.

    Parameters
    ----------
    array : np.ndarray
        Array whose values should be in the interval [0, 1] but may not be.  This array must be of
        type np.float32.
    rescale : bool
        If true, scale values by the max.

    Returns
    -------
    array : np.ndarray
        Array whose values are in the interval [0, 1].

    """
    assert array.dtype == np.float32

    array[array < 0] = 0
    if rescale:
        array /= array.max()
    else:
        array[array > 1] = 1
