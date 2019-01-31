"""
Tests for IntensityTable.from_image_stack method
"""

import numpy as np

from starfish import ImageStack, IntensityTable
from starfish.types import Axes


# TODO ambrosejcarr: crop is not tested because it should be moved out of this function
def test_intensity_table_can_be_constructed_from_an_imagestack():
    """
    ImageStack has enough information to create an IntensityTable without additional SpotAttributes.
    Each feature is a pixel, and therefore the SpotAttributes can be extracted from the relative
    locations.
    """
    r, c, z, y, x = 1, 5, 2, 2, 5
    data = np.zeros(100, dtype=np.float32).reshape(r, c, z, y, x)
    image_stack = ImageStack.from_numpy_array(data)
    intensities = IntensityTable.from_image_stack(image_stack)

    # there should be 100 features
    assert np.product(intensities.shape) == 100

    # the max features should be equal to the array extent (2, 2, 5) minus one, since indices
    # are being compared and python is zero based
    # import pdb; pdb.set_trace()
    assert np.max(intensities[Axes.ZPLANE.value].values) == z - 1
    assert np.max(intensities[Axes.Y.value].values) == y - 1
    assert np.max(intensities[Axes.X.value].values) == x - 1

    # the number of channels and rounds should match the ImageStack
    assert intensities.sizes[Axes.CH.value] == c
    assert intensities.sizes[Axes.ROUND.value] == r
