import numpy as np
import pytest

from starfish import ImageStack
from starfish.types import Number
from ..threshold import ThresholdBinarize


@pytest.mark.parametrize(["threshold"], [[threshold] for threshold in np.linspace(0, 1, 3)])
def test_binarize(threshold: Number, num_rounds=1, num_chs=1, num_zplanes=4, ysize=5, xsize=6):
    data = np.linspace(0, 1, num_rounds * num_chs * num_zplanes * ysize * xsize, dtype=np.float32)
    data = data.reshape((num_rounds, num_chs, num_zplanes, ysize, xsize))

    imagestack = ImageStack.from_numpy(data)
    binarizer = ThresholdBinarize(threshold)
    binary_mask_collection = binarizer.run(imagestack)

    assert len(binary_mask_collection) == 1
    mask = binary_mask_collection.uncropped_mask(0)

    expected_value = data[0, 0] >= threshold

    assert np.array_equal(mask, expected_value)


@pytest.mark.parametrize(
    ["num_rounds", "num_chs"],
    [
        [1, 2],
        [2, 1],
        [2, 2],
    ])
def test_binarize_non_3d(num_rounds, num_chs, num_zplanes=4, ysize=5, xsize=6):
    data = np.linspace(0, 1, num_rounds * num_chs * num_zplanes * ysize * xsize, dtype=np.float32)
    data = data.reshape((num_rounds, num_chs, num_zplanes, ysize, xsize))

    imagestack = ImageStack.from_numpy(data)
    binarizer = ThresholdBinarize(0.0)

    with pytest.raises(ValueError):
        binarizer.run(imagestack)
