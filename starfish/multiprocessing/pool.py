from multiprocessing.pool import Pool
from typing import Optional


class Spool(Pool):
    """Wrapper class for multiprocessing pool. If n_processes=1 just use map on main
    thread for debugging purposes """

    def __init__(self, processes: Optional[int]=None, **kwargs):
        Pool.__init__(self, **kwargs)
        self.n_processes = processes

    def map(self, func, iterable, chunksize=None):
        if self.n_processes == 1:
            self._initializer(*self._initargs)
            return map(func, iterable)
        return Pool.map(self, func, iterable, chunksize)

    def __reduce__(self):
        pass
