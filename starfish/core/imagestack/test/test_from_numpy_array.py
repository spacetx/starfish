import numpy as np
import pytest

from starfish.core.types import Axes, Coordinates
from ..imagestack import ImageStack


def test_from_numpy_array_raises_error_when_incorrect_dims_passed():
    array = np.ones((2, 2), dtype=np.float32)
    # verify this method works with the correct shape
    image = ImageStack.from_numpy(array.reshape((1, 1, 1, 2, 2)))
    assert isinstance(image, ImageStack)

    with pytest.raises(ValueError):
        ImageStack.from_numpy(array.reshape((1, 1, 2, 2)))
        ImageStack.from_numpy(array.reshape((1, 2, 2)))
        ImageStack.from_numpy(array)
        ImageStack.from_numpy(array.reshape((1, 1, 1, 1, 2, 2)))


def test_from_numpy_array_automatically_handles_float_conversions():
    x = np.zeros((1, 1, 1, 20, 20), dtype=np.uint16)
    stack = ImageStack.from_numpy(x)
    assert stack.xarray.dtype == np.float32


def test_from_numpy_array_coordinates():
    x = np.zeros((1, 1, 4, 20, 20), dtype=np.float32)
    coordinates = {
        Coordinates.X: np.linspace(0, 15, x.shape[-1]),
        Coordinates.Y: np.linspace(0, 20, x.shape[-2]),
        Coordinates.Z: [1, 2, 15, 20],
    }
    stack = ImageStack.from_numpy(x, coordinates=coordinates)

    assert stack.xarray.isel({Axes.X.value: 0}).coords[Coordinates.X.value] == 0
    assert stack.xarray.isel({Axes.X.value: -1}).coords[Coordinates.X.value] == 15
    assert stack.xarray.isel({Axes.Y.value: 0}).coords[Coordinates.Y.value] == 0
    assert stack.xarray.isel({Axes.Y.value: -1}).coords[Coordinates.Y.value] == 20
    assert stack.xarray.isel({Axes.ZPLANE.value: 0}).coords[Coordinates.Z.value] == 1
    assert stack.xarray.isel({Axes.ZPLANE.value: -1}).coords[Coordinates.Z.value] == 20
