import functools
import warnings
from collections import OrderedDict
from multiprocessing import Array as mp_array  # type: ignore
from typing import Callable, Sequence, Tuple

import numpy as np
import xarray as xr
from xarray import Variable

from starfish.core.types import Number


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
    def __init__(self, data: xr.DataArray, backing_mp_array: mp_array) -> None:
        self._data = data
        self._backing_mp_array = backing_mp_array

    @classmethod
    def from_shape_and_dtype(
            cls, shape: Sequence[int], dtype, initial_value: Number=None, *args, **kwargs
    ) -> "MPDataArray":
        np_array, backing_mp_array = np_array_backed_by_mp_array(shape, dtype)
        if initial_value is not None and initial_value != 0:
            np_array.fill(initial_value)
        xarray = xr.DataArray(np_array, *args, **kwargs)
        xarray.copy = functools.partial(replacement_copy, xarray.copy)
        return MPDataArray(xarray, backing_mp_array)

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
        xarray_copy, backing_mp_array_copy = xr_deepcopy(self._data)
        return MPDataArray(xarray_copy, backing_mp_array_copy)


def np_array_backed_by_mp_array(
        shape: Sequence[int], dtype) -> Tuple[np.ndarray, mp_array]:
    """Returns a np_array backed by a multiproceessing.Array buffer."""
    ctype_type = np.ctypeslib.as_ctypes(np.empty((1,), dtype=np.dtype(dtype))).__class__
    length = int(np.product(shape))  # the cast to int is required by multiprocessing.Array.
    backing_array = mp_array(ctype_type, length)
    unshaped_np_array = np.frombuffer(backing_array.get_obj(), dtype)
    shaped_np_array = unshaped_np_array.reshape(shape)

    return shaped_np_array, backing_array


def xr_deepcopy(source: xr.DataArray) -> Tuple[xr.DataArray, mp_array]:
    """Replacement for xr.DataArray's deepcopy method.  Returns a deep copy of the input xarray
    backed by a multiprocessing.Array buffer.
    """
    shaped_np_array, backing_array = np_array_backed_by_mp_array(
        source.variable.shape, source.variable.dtype)

    shaped_np_array[:] = source.variable.data

    variable = Variable(
        # dims is an immutable tuple, so it doesn't need to be deep-copyied.  see implementation of
        # Variable.__deepcopy__ for context.
        source.variable.dims,
        shaped_np_array,
        # attrs and encoding are deep copied in the constructor.
        source.variable.attrs,
        source.variable.encoding,
    )
    coords = OrderedDict((k, v.copy(deep=True))
                         for k, v in source._coords.items())

    result = xr.DataArray(
        variable,
        coords=coords,
        name=source.name,
        fastpath=True,
    )

    return result, backing_array


def replacement_copy(orig_copy: Callable, deep=True):
    if deep:
        warnings.warn(
            "Calling deepcopy with ImageStack's xarrays results in arrays that cannot be shared "
            "across processes.  Use mp_dataarray.xr_deepcopy if you need that behavior."
        )
    return orig_copy(deep)
