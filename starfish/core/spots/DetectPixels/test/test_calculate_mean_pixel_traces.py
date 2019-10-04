"""
Tests for CombineAdjacentFeatures._calculate_mean_pixel_traces method
"""

from typing import Tuple

import numpy as np
from skimage.measure import label

from starfish import ImageStack, IntensityTable
from starfish.core.spots.DetectPixels.combine_adjacent_features import CombineAdjacentFeatures
from starfish.core.types import Features


# TODO ambrosejcarr: make sure these arrays contain background pixels to catch  one-off errors
def labeled_intensities_factory() -> Tuple[IntensityTable, np.ndarray, np.ndarray]:
    """
    Create a decoded IntensityTable with distance scores, and a corresponding label_image and
    decoded_image.
    """
    data = np.array(
        [[[[0., 0.], [.1, .1]],  # ch 1
          [[.5, .5], [.2, .3]]],
         [[[.1, .1], [0, 0]],  # ch 2, x & y are reversed
          [[.2, .3], [.5, .5]]]],
        dtype=np.float32
    )
    image_stack = ImageStack.from_numpy(data.reshape(1, 2, 2, 2, 2))
    intensity_table = IntensityTable.from_image_stack(image_stack)
    intensity_table[Features.DISTANCE] = (Features.AXIS, np.zeros(intensity_table.shape[0]))
    label_image = np.array(
        [[[1, 2], [3, 4]],
         [[1, 2], [3, 4]]]
    )
    decoded_image = np.array(
        [[[5, 4], [3, 2]],
         [[5, 4], [3, 2]]]
    )

    # verify that the listed label image is what would be created by the function we use in the
    # code
    assert np.array_equal(label(decoded_image), label_image)

    return intensity_table, label_image, decoded_image


# TODO ambrosejcarr: increase test fixture complexity to verify this works over multiple rounds
def test_calculate_mean_pixel_traces():
    """
    Test that calculate_mean_pixel_traces matches the mean trace produced from the ndarray data
    used to construct the testing fixtures.
    """
    intensity_table, label_image, _ = labeled_intensities_factory()

    passes_filter = np.ones(4)
    mean_pixel_traces = CombineAdjacentFeatures._calculate_mean_pixel_traces(
        label_image,
        intensity_table,
    )

    # evaluate the mean pixel traces, we have 4 different spot ids
    assert np.unique(label_image).shape[0] == mean_pixel_traces.shape[0]

    # there should be one round and two channels
    assert np.array_equal(intensity_table.shape[1:], mean_pixel_traces.shape[1:])

    # values can be calculated from the simple example, and should match the mean pixel traces
    expected_values = np.array(
        [[[.25, .15]],
         [[.25, .2]],
         [[.15, .25]],
         [[.2, .25]]],
        dtype=np.float32
    )
    assert np.allclose(expected_values, mean_pixel_traces.values)

    # no values should be filtered, as all spots decoded
    assert np.all(passes_filter)
