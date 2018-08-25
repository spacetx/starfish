import numpy as np
import pandas as pd
from skimage.measure import regionprops

from starfish.spots._detector.combine_adjacent_features import CombineAdjacentFeatures, TargetsMap
from .test_calculate_mean_pixel_traces import labeled_intensities_factory  # reuse this fixture


# TODO what happens when things start getting filtered?
def test_create_spot_attributes():
    # make some fixtures
    intensity_table, label_image, decoded_image = labeled_intensities_factory()
    region_properties = regionprops(label_image)
    target_map = TargetsMap(np.array(list('abcdef')))
    caf = CombineAdjacentFeatures(min_area=1, max_area=3, connectivity=2)
    passes_filters = np.ones(4)
    spot_attributes, passes_filters = caf._create_spot_attributes(
        region_properties, decoded_image, target_map, passes_filters)

    # TODO this needs more rigorous testing, but seems too complicated
    assert isinstance(spot_attributes, pd.DataFrame)

    # should have a shape equal to the number of features (4)
    assert spot_attributes.shape[0] == 4

    # each feature should be correctly localized to x, y, z

    # the features should be e, d, c, b (but aren't!)

    # distances should be all 0

