"""
These tests ensure that we can share an np-array or something that encapsulates an np-array among
processes.  The data is shared such that any process can write to the array and it is visible to all
the other processes.  It is up to the developer that uses this mechanism to ensure data integrity is
maintained.
"""


import ctypes
import multiprocessing
from functools import partial
# Even though we import multiprocessing, mypy can't find the Array class.  To avoid sprinkling
# ignore markers all over the file, we explicitly import the symbol and put the ignore marker here.
from multiprocessing import Array as mp_array  # type: ignore
from typing import Any, Callable

import numpy as np

from starfish.multiprocessing import shmem


def test_numpy_array(nitems: int=10):
    """
    Try to share a numpy array directly.  This should fail as numpy is by default a copy-on-write
    object.  The worker process will write to array and we should not see the changes.
    """
    array = np.zeros(shape=(nitems,), dtype=np.uint8)
    _start_process_to_test_shmem(array, _decode_numpy_array_to_numpy_array, nitems)
    for ix in range(nitems):
        assert array[ix] == 0


def _decode_numpy_array_to_numpy_array(array):
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
    _start_process_to_test_shmem(buffer, _decode_array_to_numpy_array, nitems)
    for ix in range(nitems):
        assert array[ix] == ix


def _decode_array_to_numpy_array(array):
    return np.frombuffer(array.get_obj(), dtype=np.uint8)


class TestWrappedArray:
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
        TestWrappedArray(buffer), _decode_wrapped_array_to_numpy_array, nitems)
    for ix in range(nitems):
        assert array[ix] == ix


def _decode_wrapped_array_to_numpy_array(wrapped_array):
    return np.frombuffer(wrapped_array.array.get_obj(), dtype=np.uint8)


def _applied_func(decoder: Callable[[Any], np.ndarray], position: int) -> None:
    array = decoder(shmem.get_payload())
    array[position] = position


def _start_process_to_test_shmem(
        array_holder: Any,
        decoder: Callable[[Any], np.ndarray],
        nitems: int) -> None:
    """
    Starts a process and passes an object to its initializer.  The object is passed to
    :py:method:_applied_func, which invokes the decoder to produce a numpy array and writes to it.

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
    bound_func = partial(_applied_func, decoder)

    vals = [ix for ix in range(nitems)]
    with multiprocessing.Pool(initializer=shmem.initializer, initargs=(array_holder,)) as pool:
        pool.map(bound_func, vals)
