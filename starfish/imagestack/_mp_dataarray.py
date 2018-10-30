import copy
from typing import Sequence

import numpy as np
import xarray as xr

from starfish.types import Number


class MPDataArray:
    """Wrapper class for xarray.  It provides limited delegation to simplify ImageStack, but code
    external to ImageStack should transact in the enclosed xarray, as operations that involve
    special method names (e.g., __eq__) do not delegate correctly.

    This is necessary for us to stack an xarray on top of a numpy array that is stacked on top of a
    multiprocessing.Array object.  When we want to pass the xarray to worker processes using
    Python's multiprocessing module, we need to pass the underlying multiprocessing.Array object
    instead of the xarray or the numpy array.  However, there is no way to extract the
    multiprocessing.Array object back out of the numpy array or the xarray.  Therefore, we need to
    explicitly maintain a reference to it and keep the two items together.
    """
    def __init__(self, data: xr.DataArray) -> None:
        self._data = data

    @classmethod
    def from_shape_and_dtype(
            cls, shape: Sequence[int], dtype, initial_value: Number=None, *args, **kwargs
    ) -> "MPDataArray":
        np_array = np.empty(shape=shape, dtype=dtype)
        if initial_value is not None:
            np_array.fill(initial_value)
        xarray = xr.DataArray(np_array, *args, **kwargs)
        return MPDataArray(xarray)

    @property
    def data(self) -> xr.DataArray:
        return self._data

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __deepcopy__(self, memodict={}):
        return MPDataArray(copy.deepcopy(self._data, memodict))
