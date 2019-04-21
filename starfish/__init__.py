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
# display images and spots
from ._display import display
# generated version number and commit hash
from ._version import get_versions
# top-level objects
from ._codebook.codebook import Codebook
from ._experiment.experiment import Experiment, FieldOfView
from ._imagestack.imagestack import ImageStack
from ._intensity_table.intensity_table import IntensityTable
from ._starfish import starfish


# NOTE: if we move to python 3.7, we can produce this value at call time via __getattr__
__version__ = get_versions()['version']
__is_release_tag__ = None
if "+" not in str(__version__):
    __is_release_tag__ = f"Release: {__version__}"
del get_versions

if __name__ == "__main__":
    starfish()

del warnings, pkg_resources
