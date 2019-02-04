# deal with numpy import warnings due to cython
# See: https://stackoverflow.com/questions/40845304/
#      runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility)
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # noqa
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # noqa

import pkg_resources

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


# NOTE: if we move to python 3.7, we can produce this value at call time via __getattr__
__version__ = pkg_resources.require("starfish")[0].version


if __name__ == "__main__":
    starfish()
