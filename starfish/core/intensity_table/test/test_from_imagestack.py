"""
Tests for IntensityTable.from_image_stack method
"""

import numpy as np

from starfish import ImageStack
from starfish.core.imagestack.test import test_labeled_indices
from starfish.core.test.factories import (
    codebook_intensities_image_for_single_synthetic_spot,
    synthetic_spot_pass_through_stack,
)
from starfish.core.types import Axes
from ..intensity_table import IntensityTable


def test_imagestack_to_intensity_table():
    codebook, intensity_table, image = codebook_intensities_image_for_single_synthetic_spot()
    pixel_intensities = IntensityTable.from_image_stack(image)
    pixel_intensities = codebook.decode_metric(
        pixel_intensities, max_distance=0, min_intensity=1000, norm_order=2)
    assert isinstance(pixel_intensities, IntensityTable)


def test_imagestack_to_intensity_table_no_noise():
    codebook, intensity_table, image = synthetic_spot_pass_through_stack()
    pixel_intensities = IntensityTable.from_image_stack(image)
    pixel_intensities = codebook.decode_metric(
        pixel_intensities, max_distance=0, min_intensity=1000, norm_order=2)
    assert isinstance(pixel_intensities, IntensityTable)


def test_intensity_table_can_be_constructed_from_an_imagestack():
    """
    ImageStack has enough information to create an IntensityTable without additional SpotAttributes.
    Each feature is a pixel, and therefore the SpotAttributes can be extracted from the relative
    locations.
    """
    r, c, z, y, x = 1, 5, 2, 2, 5
    data = np.zeros(100, dtype=np.float32).reshape(r, c, z, y, x)
    image_stack = ImageStack.from_numpy(data)
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


def test_from_imagestack_labeled_indices():
    # use the ImageStack with labeled indices from the test.
    imagestack = test_labeled_indices.setup_imagestack()
    intensity_table = IntensityTable.from_image_stack(imagestack)
    assert np.array_equal(
        intensity_table[Axes.CH.value], np.array(test_labeled_indices.CH_LABELS))
    assert np.array_equal(
        intensity_table[Axes.ROUND.value], np.array(test_labeled_indices.ROUND_LABELS))
