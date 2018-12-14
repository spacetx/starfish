# deal with numpy import warnings due to cython
# See: https://stackoverflow.com/questions/40845304/
#      runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility)
import mmap
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # noqa
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # noqa
from multiprocessing.heap import Arena

# image processing methods and objects
from . import image
# spot detection and manipulation
from . import spots
# top-level objects
from .codebook.codebook import Codebook
from .experiment.experiment import Experiment, FieldOfView
from .imagestack.imagestack import ImageStack
from .intensity_table.intensity_table import IntensityTable
from .starfish import starfish


def anonymous_arena_init(self, size, fd=-1):
    "Create Arena using an anonymous memory mapping."
    self.size = size
    self.fd = fd  # still kept but is not used !
    self.buffer = mmap.mmap(-1, self.size)


Arena.__init__ = anonymous_arena_init

if __name__ == "__main__":
    starfish()
