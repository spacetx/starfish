from collections import OrderedDict

import pytest
from slicedimage import ImageFormat

from starfish.core.types import Axes, PhysicalCoordinateTypes
from .factories import imagestack_with_coords_factory
from .imagestack_test_utils import verify_physical_coordinates
from ..imagestack import ImageStack
from ..physical_coordinates import _get_physical_coordinates_of_z_plane

X_COORDS = 1, 2
Y_COORDS = 4, 6
Z_COORDS = 1, 3


@pytest.mark.parametrize("format,count", (
    (ImageFormat.TIFF, 6),
    (ImageFormat.NUMPY, 6),
))
def test_imagestack_export(tmpdir, format, count, recwarn):
    """
    Save a synthetic stack to files and check the results
    """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 50), (Axes.X, 40)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)

    stack_json = tmpdir / "output.json"
    stack.export(str(stack_json), tile_format=format)
    files = list([x for x in tmpdir.listdir() if str(x).endswith(format.file_ext)])
    loaded_stack = ImageStack.from_path_or_url(str(stack_json))
    verify_physical_coordinates(
        loaded_stack,
        X_COORDS,
        Y_COORDS,
        _get_physical_coordinates_of_z_plane(Z_COORDS),
    )
    assert count == len(files)
    with open(files[0], "rb") as fh:
        format.reader_func(fh)
