import multiprocessing.pool as mp
from typing import Optional


class Pool(mp.Pool):
    """Wrapper class for multiprocessing pool. If n_processes=1 just use map on main
    thread for debugging purposes """

    def __init__(self, processes: Optional[int]=None, *args, **kwargs):
        mp.Pool.__init__(self, processes, *args, **kwargs)
        self.n_processes = processes

    def map(self, func, iterable, chunksize=None):
        if self.n_processes == 1:
            self._initializer(*self._initargs)
            return map(func, iterable)
        return mp.Pool.map(self, func, iterable, chunksize)

    def imap(self, func, iterable, chunksize=1):
        if self.n_processes == 1:
            self._initializer(*self._initargs)
            return map(func, iterable)
        return mp.Pool.imap(self, func, iterable, chunksize)

