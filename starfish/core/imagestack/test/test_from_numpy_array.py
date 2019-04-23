import numpy as np
import pytest

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
