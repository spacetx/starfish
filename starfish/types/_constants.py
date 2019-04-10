from enum import Enum


class AugmentedEnum(Enum):
    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, type(self)) or isinstance(other, str):
            return self.value == other
        return False

    def __str__(self) -> str:
        return self.value


class Coordinates(AugmentedEnum):
    Z = 'zc'
    Y = 'yc'
    X = 'xc'


PHYSICAL_COORDINATE_DIMENSION = "physical_coordinate"

"""
This is the xarray dimension name for the physical coordinates of the tiles.
"""

STARFISH_EXTRAS_KEY = 'starfish'
"""
Attribute on Imagestack and IntensityTable for storing starfish related info
"""
LOG = "log"
"""
This is name of the provenance log attribute stored on the IntensityTable
"""
CORE_DEPENDENCIES = {'numpy', 'scikit-image', 'pandas', 'scikit-learn', 'scipy', 'xarray', 'sympy'}
"""
The set of dependencies whose versions are are logged for each starfish session
"""


class PhysicalCoordinateTypes(AugmentedEnum):
    """
    These are more accurately the xarray coordinates for the physical coordinates of a tile.
    """
    Z_MAX = 'zmax'
    Z_MIN = 'zmin'
    Y_MAX = 'ymax'
    Y_MIN = 'ymin'
    X_MAX = 'xmax'
    X_MIN = 'xmin'


class Axes(AugmentedEnum):
    ROUND = 'r'
    CH = 'c'
    ZPLANE = 'z'
    Y = 'y'
    X = 'x'


class Features:
    """
    contains constants relating to the codebook and feature (spot/pixel) representations of the
    image data
    """

    AXIS = 'features'
    TARGET = 'target'
    CODEWORD = 'codeword'
    CODE_VALUE = 'v'
    SPOT_RADIUS = 'radius'
    DISTANCE = 'distance'
    PASSES_THRESHOLDS = 'passes_thresholds'
    CELL_ID = 'cell_id'
    SPOT_ID = 'spot_id'
    INTENSITY = 'intensity'
    AREA = 'area'
    CELLS = 'cells'
    GENES = 'genes'


class OverlapStrategy(AugmentedEnum):
    """
    contains options to use when processes physically overlapping IntensityTables
    or ImageStacks
    """
    TAKE_MAX = 'take_max'


class Clip(AugmentedEnum):
    """
    contains clipping options that determine how out-of-bounds values produced by filters are
    treated to keep the image contained within [0, 1]
    """
    CLIP = 'clip'
    SCALE_BY_IMAGE = 'scale_by_image'
    SCALE_BY_CHUNK = 'scale_by_chunk'


class TransformType(AugmentedEnum):
    """
    currently supported transform types
    """
    SIMILARITY = 'similarity'
