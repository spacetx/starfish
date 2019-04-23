import mmap
from multiprocessing.heap import Arena
from sys import platform


def anonymous_arena_init(self, size, fd=-1):
    """
    Patching multiprocessing.heap.Arena.__init__ so that it uses anonymous memory mapping
    instead of creating a temp file for shared memory amongst subprocess.

    fix from: https://bugs.python.org/issue30919
    """
    self.size = size
    self.fd = fd  # still kept but is not used !
    self.buffer = mmap.mmap(-1, self.size)


if platform.startswith("darwin"):
    # Only do in Mac env.
    Arena.__init__ = anonymous_arena_init
