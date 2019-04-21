"""
Because multiprocessing.Pool demands that any shared-memory constructs be passed in as initializer
arguments, and not as arguments to multiprocessing.Pool.map or multiprocessing.Pool.apply, we need a
stub initializer that accepts the shared-memory construct and store it in a global for retrieval.

Does not work:
  arrays = [multiprocessing.Array(..)]
  with multiprocessing.Pool() as pool:
    pool.map(some_func, arrays)

Does work:
  array = multiprocessing.Array(..)
  data = range(10)
  with multiprocessing.Pool(initializer=some_function, initargs=(array,)) as pool:
    pool.map(some_func, data)

This wraps the ugliness of a global variable inside a class that's only used for this purpose.  An
example of how this might be used would be:

  def worker(val):
    array = SharedMemory.get_payload()
    # do something with the shared memory

  array = multiprocessing.Array(..)
  data = range(10)
  with multiprocessing.Pool(initializer=SharedMemory.initalizer, initargs=(array,)) as pool:
    pool.map(worker, data)
"""
from typing import Any


class SharedMemory:
    _payload = None

    @staticmethod
    def initializer(payload: Any) -> None:
        SharedMemory._payload = payload

    @staticmethod
    def get_payload() -> Any:
        return SharedMemory._payload
