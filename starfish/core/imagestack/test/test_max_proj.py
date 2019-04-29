import numpy as np

from starfish import data
from starfish.core.types import Axes
from ..imagestack import ImageStack


def test_max_projection_preserves_dtype():
    original_dtype = np.float32
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy(array.reshape((1, 1, 2, 2, 2)))

    max_projection = image.max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
    assert max_projection.xarray.dtype == original_dtype


def test_max_projection_preserves_coordinates():
    e = data.ISS(use_test_data=True)
    nuclei = e.fov().get_image('nuclei')
    nuclei_proj = nuclei.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    # Since this data already has only 1 round, 1 ch, 1 zplane
    # let's just assert that the max_proj operation didn't change anything
    assert nuclei.xarray.equals(nuclei_proj.xarray)
