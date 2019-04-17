import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes


def test_max_projection_preserves_dtype():
    original_dtype = np.float32
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2, 2)))

    max_projection = image.max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
    assert max_projection.xarray.dtype == original_dtype
