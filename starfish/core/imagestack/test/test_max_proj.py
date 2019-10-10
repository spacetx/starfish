from collections import OrderedDict

import numpy as np

from starfish import data
from starfish.core.types import Axes, PhysicalCoordinateTypes
from .factories import imagestack_with_coords_factory
from .imagestack_test_utils import verify_physical_coordinates
from ..imagestack import ImageStack


def test_max_projection_preserves_dtype():
    original_dtype = np.float32
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy(array.reshape((1, 1, 2, 2, 2)))

    max_projection = image.reduce((Axes.CH, Axes.ROUND, Axes.ZPLANE), "max")
    assert max_projection.xarray.dtype == original_dtype


X_COORDS = 1, 2
Y_COORDS = 4, 6
Z_COORDS = 1, 3


def test_max_projection_preserves_coordinates():
    e = data.ISS(use_test_data=True)
    nuclei = e.fov().get_image('nuclei')
    nuclei_proj = nuclei.reduce((Axes.CH, Axes.ROUND, Axes.ZPLANE), "max")
    # Since this data already has only 1 round, 1 ch, 1 zplane
    # let's just assert that the max_proj operation didn't change anything
    assert nuclei.xarray.equals(nuclei_proj.xarray)

    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 3), (Axes.Y, 10), (Axes.X, 10)])

    # Create stack with coordinates, verify coords unaffected by max_poj
    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)

    stack_proj = stack.reduce((Axes.CH, Axes.ROUND, Axes.ZPLANE), "max")
    expected_z = np.average(Z_COORDS)
    verify_physical_coordinates(stack_proj, X_COORDS, Y_COORDS, expected_z)
