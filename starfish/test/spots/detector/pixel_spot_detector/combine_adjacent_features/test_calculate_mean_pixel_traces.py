import numpy as np
from skimage.measure import label

from starfish import IntensityTable, ImageStack
from starfish.spots._detector.combine_adjacent_features import CombineAdjacentFeatures
from starfish.types import Features


def labeled_intensities_factory():
    data = np.array(
        [[[[0, 0], [1, 1]],  # ch 1
          [[5, 5], [2, 3]]],
         [[[1, 1], [0, 0]],  # ch 2, x & y are reversed
          [[2, 3], [5, 5]]]]

    )
    image_stack = ImageStack.from_numpy_array(data.reshape(1, 2, 2, 2, 2))
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


# todo do I need to make sure this works over multiple hybridization rounds?
def test_calculate_mean_pixel_traces():
    intensity_table, label_image, _ = labeled_intensities_factory()

    passes_filter = np.ones(4)
    mean_pixel_traces, passes_filter = CombineAdjacentFeatures._calculate_mean_pixel_traces(
        label_image,
        intensity_table,
        passes_filter
    )

    # evaluate the mean pixel traces, we have 4 different spot ids
    assert np.unique(label_image).shape[0] == mean_pixel_traces.shape[0]

    # there should be two channels and 1 round
    assert np.array_equal(intensity_table.shape[1:], mean_pixel_traces.shape[1:])

    # values can be calculated from the simple example, and should match the mean pixel traces
    expected_values = np.array(
        [[[2.5],
          [1.5]],
         [[2.5],
          [2]],
         [[1.5],
          [2.5]],
         [[2],
          [2.5]]]
    )
    assert np.array_equal(expected_values, mean_pixel_traces.values)

    # no values should be filtered, as all spots decoded
    assert np.all(passes_filter)

