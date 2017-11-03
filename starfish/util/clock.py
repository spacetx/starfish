import contextlib
import time


@contextlib.contextmanager
def timeit(callback):
    before = time.time()
    yield
    after = time.time()
    callback(after - before)
