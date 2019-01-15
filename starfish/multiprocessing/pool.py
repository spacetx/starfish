from multiprocessing.pool import Pool
from typing import Optional


class StarfishPool(Pool):
    """Wrapper class for multiprocessing pool. When numper of processes = 1...."""

    def __init__(self, processes: Optional[int]=None, **kwargs):
        Pool.__init__(self, **kwargs)
        self.n_processes = processes

    def map(self, func, iterable, chunksize=None):
        if self.n_processes == 1:
            return map(func, iterable)
        return Pool.map(self, func, iterable, chunksize)

    def __reduce__(self):
        pass
