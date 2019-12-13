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


class Levels(AugmentedEnum):
    """
    Controls the way that data are scaled to retain skimage dtype requirements that float data fall
    in [0, 1].  In all modes, data below 0 are set to 0.
    """
    CLIP = "clip"
    """Data above 1 are set to 1."""
    SCALE_SATURATED_BY_IMAGE = 'scale_saturated_by_image'
    """If peak intensity of the entire image is saturated (i.e., > 1), rescale the intensity of the
    entire image by the peak intensity.  If peak intensity of the entire image is not saturated
    (i.e., <= 1), do not rescale.  This is functionally equivalent to Clip.SCALE_BY_IMAGE."""
    SCALE_SATURATED_BY_CHUNK = 'scale_saturated_by_chunk'
    """If the peak intensity of an image chunk is saturated (i.e., > 1), rescale the intensity of
    the chunk by the peak intensity.  If peak intensity of an image chunk is not saturated
    (i.e., <= 1), do not rescale.  This is functionally equivalent to Clip.SCALE_BY_CHUNK."""
    SCALE_BY_IMAGE = 'scale_by_image'
    """Rescale the intensity of the entire image by the peak intensity.  Note that if the peak
    intensity of the entire image is not saturated, this behaves differently than
    Clip.SCALE_BY_IMAGE."""
    SCALE_BY_CHUNK = 'scale_by_chunk'
    """Rescale the intensity of an image chunk by the peak intensity.  Note that if the peak
    intensity of an image chunk is not saturated, this behaves differently than
    Clip.SCALE_BY_CHUNK."""


class TransformType(AugmentedEnum):
    """
    currently supported transform types
    """
    SIMILARITY = 'similarity'


class TraceBuildingStrategies(AugmentedEnum):
    """
    currently support spot trace building strategies
    """
    EXACT_MATCH = 'exact_match'
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    SEQUENTIAL = 'sequential'
