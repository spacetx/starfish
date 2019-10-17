from typing import MutableSequence, Sequence, Union

import numpy as np
import xarray as xr

from starfish.types import Axes
from .util import _get_axes_names, AXES_ORDER


def fill_from_mask(
        mask: xr.DataArray,
        fill_value: int,
        result_array: np.ndarray,
        axes_order: Sequence[Union[str, Axes]] = AXES_ORDER,
):
    """Take a binary mask with labeled axes and write `fill_value` to an array `result_array` where
    the binary mask has a True value.  The output array is assumed to have a zero origin.  The input
    mask has unspecified axes orders and the axes labels are used to determine the origin.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> mask = xr.DataArray([True, True, False], dims='x', coords={'x': [1, 2, 3]})
    >>> mask
    <xarray.DataArray (x: 3)>
    array([ True,  True, False])
    Coordinates:
      * x        (x) int64 1 2 3
    >>> result_array = np.zeros(shape=(4,), dtype=np.uint32)
    >>> fill_from_mask(mask, 2, result_array)
    >>> result_array
    array([0, 2, 2, 2], dtype=uint32)
    """
    axes_order = tuple(axis.value if isinstance(axis, Axes) else axis for axis in axes_order)
    axes, _ = _get_axes_names(mask.ndim)
    selector: MutableSequence[slice] = []
    for axis_number, axis in enumerate(axes_order):
        coord_values = mask.coords[axis].values
        selector.append(slice(coord_values[0], coord_values[-1] + 1))
        if coord_values[0] < 0:
            raise ValueError(
                f"labels for axis {axis} should range from 0 to "
                f"{result_array.shape[axis_number] - 1}.  The minimum value found was "
                f"{coord_values[0]}"
            )
        if coord_values[-1] >= result_array.shape[axis_number]:
            raise ValueError(
                f"labels for axis {axis} should range from 0 to "
                f"{result_array.shape[axis_number] - 1}.  The maximum value found was "
                f"{coord_values[-1]}"
            )

    fill_value_array = result_array[selector]
    fill_value_array[mask.values] = fill_value
    result_array[selector] = fill_value_array
