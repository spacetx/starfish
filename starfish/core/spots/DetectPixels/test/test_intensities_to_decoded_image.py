"""
Tests for CombineAdjacentFeatures._intensities_to_decoded_image method
"""

from typing import Tuple

import numpy as np

from starfish import ImageStack, IntensityTable
from starfish.core.spots.DetectPixels.combine_adjacent_features import (
    CombineAdjacentFeatures, TargetsMap
)
from starfish.core.types import Features


def decoded_intensity_table_factory() -> Tuple[IntensityTable, np.ndarray]:
    """
    Create an IntensityTable that has gene labels, including null labels. The data doesn't matter,
    so will use np.zeros
    """
    data = np.zeros((1, 1, 2, 3, 3), dtype=np.float32)
    labels = np.array(
        [[[0, 1, 1],
          [0, 2, 2],
          [1, 1, 1]],
         [[0, 1, 1],
          [1, 1, 1],
          [0, 1, 2]]],
        dtype='<U3'
    )
    labels_with_nan = labels.copy()
    labels_with_nan[labels == '0'] = 'nan'

    # create an intensity table and add the labels
    image_stack = ImageStack.from_numpy(data)
    intensities = IntensityTable.from_image_stack(image_stack)
    intensities[Features.TARGET] = (Features.AXIS, np.ravel(labels_with_nan))

    # label the third column of this data as failing filters
    passes_filters = np.ones(data.shape, dtype=bool)
    passes_filters[:, :, :, :, -1] = 0
    intensities[Features.PASSES_THRESHOLDS] = (Features.AXIS, np.ravel(passes_filters))

    return intensities, labels_with_nan


def test_intensities_to_decoded_image():
    """Test that the decoded image matches the labels object that it is expected to."""

    intensities, labels_with_nan = decoded_intensity_table_factory()

    # test producing a decoded image
    targets_map = TargetsMap(intensities[Features.TARGET].values)
    decoded_image = CombineAdjacentFeatures._intensities_to_decoded_image(
        intensities,
        targets_map,
        mask_filtered_features=False
    )

    # we should be able to map the results back to labels_with_nan using the targets dict
    reverted_labels = targets_map.targets_as_str(np.ravel(decoded_image)).reshape(2, 3, 3)
    assert np.array_equal(reverted_labels, labels_with_nan)


def test_intensities_failing_filters_are_masked_when_requested():
    """
    Test that a masking request causes masked features to be set to zero (background) in the label
    image.
    """

    intensities, labels_with_nan = decoded_intensity_table_factory()

    # test producing a decoded image
    targets_map = TargetsMap(intensities[Features.TARGET].values)
    decoded_image = CombineAdjacentFeatures._intensities_to_decoded_image(
        intensities,
        targets_map,
        mask_filtered_features=True
    )

    # the third column should now be masked
    assert np.all(decoded_image[:, :, -1] == 0)
