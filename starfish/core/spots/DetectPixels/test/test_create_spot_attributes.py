"""
Tests for CombineAdjacentFeatures._create_spot_attributes method
"""

import numpy as np
from skimage.measure import regionprops

from starfish.core.spots.DetectPixels.combine_adjacent_features import (
    CombineAdjacentFeatures, TargetsMap
)
from starfish.core.types import Axes, Features, SpotAttributes
from .test_calculate_mean_pixel_traces import labeled_intensities_factory  # reuse this fixture


def test_create_spot_attributes():
    """
    Verify that SpotAttributes are correctly created by verifying that a simple example maps each
    feature to the expected x, y, z coordinates and that the correct targets are associated with
    each feature
    """
    # make some fixtures
    intensity_table, label_image, decoded_image = labeled_intensities_factory()
    region_properties = regionprops(np.squeeze(label_image))
    target_map = TargetsMap(np.array(list('abcdef')))
    caf = CombineAdjacentFeatures(min_area=1, max_area=3, connectivity=2)
    spot_attributes, passes_filters = caf._create_spot_attributes(
        region_properties, decoded_image, target_map
    )

    assert isinstance(spot_attributes, SpotAttributes)

    # should have a shape equal to the number of features (4)
    assert spot_attributes.data.shape[0] == 4

    # each feature should be correctly localized to z, y, x, and the SpotAttributes should be
    # sorted in that order
    assert np.array_equal(spot_attributes.data[Axes.ZPLANE], [0, 0, 0, 0])
    assert np.array_equal(spot_attributes.data[Axes.Y], [0, 0, 1, 1])
    assert np.array_equal(spot_attributes.data[Axes.X], [0, 1, 0, 1])

    # spots should map to edcb -- the decoded image has 4 values: [5, 4, 3, 2]. The target map
    # starts at np.nan, then counts sequentially from 1. Thus, 2 should map to b, and from there
    # the values are sequential. f=6 is not present.
    assert np.array_equal(spot_attributes.data[Features.TARGET].values, list('edcb'))
