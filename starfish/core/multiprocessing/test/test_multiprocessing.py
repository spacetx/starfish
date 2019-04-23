"""
These tests ensure that we can share an np-array or something that encapsulates an np-array among
processes.  The data is shared such that any process can write to the array and it is visible to all
the other processes.  It is up to the developer that uses this mechanism to ensure data integrity is
maintained.
"""
import copy
import ctypes
import multiprocessing
import warnings
from functools import partial
# Even though we import multiprocessing, mypy can't find the Array class.  To avoid sprinkling
# ignore markers all over the file, we explicitly import the symbol and put the ignore marker here.
from multiprocessing import Array as mp_array  # type: ignore
from typing import Any, Callable, Tuple

import numpy as np
import xarray as xr

from starfish import ImageStack
from starfish.core.imagestack import _mp_dataarray
from ..shmem import SharedMemory


def test_numpy_array(nitems: int=10):
    """
    Try to share a numpy array directly.  This should fail as numpy is by default a copy-on-write
    object.  The worker process will write to array and we should not see the changes.
    """
    array = np.zeros(shape=(nitems,), dtype=np.uint8)
    _start_process_to_test_shmem(
        array_holder=array,
        decoder=_decode_numpy_array_to_numpy_array,
        nitems=nitems)
    for ix in range(nitems):
        assert array[ix] == 0


def _decode_numpy_array_to_numpy_array(array):
    """
    When passing naked numpy arrays into subprocesses, no work is needed to retrieve the array. See
    :py:method:`_decode_array_to_numpy_array` for an example where decoding is necessary.
    """
    return array


def test_shmem_numpy_array(nitems: int=10):
    """
    Try to share a numpy array based on a multiprocessing Array object.  The array object is passed
    to the worker process.  The worker process reconstitutes the numpy array from that memory buffer
    and writes to the reconstituted numpy array.  Writes in the worker process should be visible in
    the parent process.
    """
    buffer = mp_array(ctypes.c_uint8, nitems)
    array = _decode_array_to_numpy_array(buffer)
    array.fill(0)
    _start_process_to_test_shmem(
        array_holder=buffer,
        decoder=_decode_array_to_numpy_array,
        nitems=nitems)
    for ix in range(nitems):
        assert array[ix] == ix


def _decode_array_to_numpy_array(array):
    return np.frombuffer(array.get_obj(), dtype=np.uint8)


class WrappedArray:
    """
    Dummy class that wraps a multiprocessing array in an object.
    """
    def __init__(self, array: mp_array) -> None:
        self.array = array


def test_wrapped_shmem_numpy_array(nitems: int=10):
    """
    Try to share a numpy array based on a multiprocessing Array object.  The array object is wrapped
    in a container class and passed to the worker process.  The worker process reconstitutes the
    numpy array from that memory buffer and writes to the reconstituted numpy array.  Writes in the
    worker process should be visible in the parent process.
    """
    buffer = mp_array(ctypes.c_uint8, nitems)
    array = _decode_array_to_numpy_array(buffer)
    array.fill(0)
    _start_process_to_test_shmem(
        array_holder=WrappedArray(buffer),
        decoder=_decode_wrapped_array_to_numpy_array,
        nitems=nitems)
    for ix in range(nitems):
        assert array[ix] == ix


def _decode_wrapped_array_to_numpy_array(wrapped_array: WrappedArray) -> np.ndarray:
    return np.frombuffer(wrapped_array.array.get_obj(), dtype=np.uint8)


def test_xr_deepcopy(nitems: int=10) -> None:
    """
    Instantiate an :py:class:`xarray.DataArray` and run
    :py:method:`starfish.imagestack.mp_dataarray.xr_deepcopy` on it.  The copy is passed to worker
    processes.  Worker processes reconstitute a numpy array from the buffer and attempts to writes
    to the numpy array.  Writes in the worker process should be visible in the parent process in the
    copy but not the original.
    """
    shape = (nitems, )
    source = np.zeros(shape, dtype=np.uint8)
    dataarray = xr.DataArray(source)
    copy, buffer = _mp_dataarray.xr_deepcopy(dataarray)
    _start_process_to_test_shmem(
        array_holder=buffer,
        decoder=_decode_array_to_numpy_array,
        nitems=nitems)
    for ix in range(nitems):
        assert dataarray[ix] == 0
        assert copy[ix] == ix


def test_imagestack_deepcopy(nitems: int=10) -> None:
    """
    Instantiate an :py:class:`ImageStack` and deepcopy it.  Worker processes reconstitute a numpy
    array from the buffer and attempts to writes to the numpy array.  Writes in the worker process
    should be visible in the parent process.
    """
    shape = (nitems, 3, 4, 5, 6)
    dtype = np.float32
    source = np.zeros(shape, dtype=np.float32)
    imagestack = ImageStack.from_numpy(source)
    imagestack_copy = copy.deepcopy(imagestack)
    _start_process_to_test_shmem(
        array_holder=imagestack_copy._data._backing_mp_array,
        decoder=partial(_decode_imagestack_array_to_numpy_array, shape, dtype),
        nitems=nitems)
    for ix in range(nitems):
        assert (imagestack.xarray[ix] == 0).all()
        assert np.allclose(imagestack_copy.xarray[ix], ix)


def _decode_imagestack_array_to_numpy_array(
        shape: Tuple[int, ...], dtype, buffer) -> np.ndarray:
    unshaped_np_array = np.frombuffer(buffer.get_obj(), dtype=dtype)
    return unshaped_np_array.reshape(shape)


def test_imagestack_xarray_deepcopy(nitems: int=10) -> None:
    """
    Instantiate an :py:class:`ImageStack` and deepcopy the xarray directly.  This should work, but
    prompt a warning.
    """
    shape = (nitems, 3, 4, 5, 6)
    dtype = np.float32
    source = np.zeros(shape, dtype=dtype)
    imagestack = ImageStack.from_numpy(source)
    with warnings.catch_warnings(record=True) as warnings_:
        copy.deepcopy(imagestack.xarray)
        assert len(warnings_) == 1  # type: ignore


def _write_to_shared_memory_array(decoder: Callable[[Any], np.ndarray], position: int) -> None:
    array = decoder(SharedMemory.get_payload())
    array[position] = position


def _start_process_to_test_shmem(
        *,
        array_holder: Any,
        decoder: Callable[[Any], np.ndarray],
        nitems: int) -> None:
    """
    This method leverages multiprocessing.Pool to test process initializers and shared memory array
    decoders.

    Starts a process Pool and passes :py:method:`SharedMemory.initializer` as its initializer, which
    will store `array_holder` as a global variable inside the initialized process.

    Creates `bound_func`, which will be executed by the subprocesses of Pool.  `bound_func` is a
    partial function constructed from :py:method:`_write_to_shared_memory_array` that exposes
    `decoder`, a method that gets a numpy array out of array_holder and then writes a value to it
    (inside the spawned process).

    This enables writing of tests that check whether the written values are available in the parent
    processes, which depends on `SharedMemory.initializer`, `array_holder`, and `decoder`.

    All the subprocesses must complete before this method returns.

    Parameters
    ----------
    array_holder : Any
        An object that when passed to the callable specified in `decoder`.  This is the object that
        is passed in :py:class:`multiprocessing.Pool`'s initargs.
    decoder : Callable[[Any], np.ndarray]
        A callable that takes the object in `array_holder` and returns a np.ndarray.
    nitems : int
        Number of jobs to spawn.  Each job will write a single element of the array.
    """
    bound_func = partial(_write_to_shared_memory_array, decoder)

    vals = [ix for ix in range(nitems)]
    with multiprocessing.Pool(
            initializer=SharedMemory.initializer, initargs=(array_holder,)) as pool:
        pool.map(bound_func, vals)
