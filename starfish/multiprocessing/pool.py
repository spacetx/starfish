import multiprocessing.pool as mp
import os
from typing import Callable, Iterable, Optional


class Pool:
    """Wrapper class for multiprocessing pool. If n_processes=1 just use map on main
    thread for debugging purposes """

    def __init__(
            self,
            processes: Optional[int]=None,
            initializer: Optional[Callable]=None,
            initargs: Optional[Iterable]=None,
            *args, **kwargs):
        self.initializer = initializer
        self.initargs = initargs or []
        if processes == 1 or os.name == "nt":
            self.pool = None
        else:
            self.pool = mp.Pool(processes, self.initializer, self.initargs, *args, **kwargs)

    def map(self, func, iterable, chunksize=None):
        if self.pool is None:
            self.initializer(*self.initargs)
            return map(func, iterable)
        return self.pool.map(func, iterable, chunksize)

    def imap(self, func, iterable, chunksize=1):
        if self.pool is None:
            self.initializer(*self.initargs)
            return map(func, iterable)
        return self.pool.imap(func, iterable, chunksize)

    def __enter__(self, *args, **kwargs):
        if self.pool is None:
            return self
        return self.pool.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        if self.pool is None:
            return None
        return self.pool.__exit__(*args, **kwargs)
