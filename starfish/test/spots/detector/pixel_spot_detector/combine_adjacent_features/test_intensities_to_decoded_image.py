import numpy as np

from starfish import ImageStack, IntensityTable
from starfish.types import Features
from starfish.spots._detector.combine_adjacent_features import TargetsMap, CombineAdjacentFeatures


def test_intensities_to_decoded_image():

    # mock up an ImageStack that has gene labels, including null labels
    # data doesn't matter
    data = np.zeros((1, 1, 2, 3, 3))
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
    image_stack = ImageStack.from_numpy_array(data)
    intensities = IntensityTable.from_image_stack(image_stack)
    intensities[Features.TARGET] = (Features.AXIS, np.ravel(labels_with_nan))

    # test producing a decoded image
    targets_map = TargetsMap(intensities[Features.TARGET].values)
    decoded_image = CombineAdjacentFeatures._intensities_to_decoded_image(intensities, targets_map)

    # because we've mutated zero to nan, and we otherwise map numbers sequentially, this decoded
    # outcome should be the same as the input labels, converted to ints.
    # int_labels = labels.astype(int)
    # assert np.array_equal(int_labels, decoded_image)

    # TODO there is some rearranging going on here?
    # we should be able to map the results back to labels_with_nan using the targets dict
    reverted_labels = targets_map.targets_as_str(np.ravel(decoded_image)).reshape(2, 3, 3)
    assert np.array_equal(reverted_labels, labels_with_nan)
