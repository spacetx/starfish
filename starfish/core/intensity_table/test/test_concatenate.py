import numpy as np

from starfish import ImageStack, IntensityTable
from ..concatenate import concatenate


def test_intensity_table_concatenation():
    """create two IntensityTables and assert that they are being concatenated properly."""

    r, c, z, y, x = 3, 3, 2, 2, 5
    data = np.zeros(180, dtype=np.float32).reshape(r, c, z, y, x)
    image_stack = ImageStack.from_numpy(data)
    intensities = IntensityTable.from_image_stack(image_stack)

    intensities2 = intensities.copy()

    original_shape = intensities.shape

    expected_shape = list(original_shape)
    expected_shape[0] *= 2  # only features is concatenated
    assert np.array_equal(
        concatenate([intensities, intensities2]).shape,
        expected_shape
    )

    # slice out a single channel and round from both experiments, such that the data no longer match
    # across all dimensions but the concatenation dimension. The resulting structure should be
    # 2 (r) * 2 (c) * 5 (z), 2, 2 = 40, 2, 2
    i1 = intensities.where(np.logical_and(intensities.r == 0, intensities.c == 0), drop=True)
    i2 = intensities.where(np.logical_and(intensities.r == 1, intensities.c == 1), drop=True)
    expected_shape = (i1.shape[0] + i2.shape[0], 2, 2)
    result = concatenate([i1, i2])

    assert expected_shape == result.shape

    # slice a larger r value for second array, however, there are still only two values, so
    # shape should be 40, 2, 2
    i3 = intensities.where(np.logical_and(intensities.r == 2, intensities.c == 1), drop=True)
    expected_shape = (i1.shape[0] + i3.shape[0], 2, 2)
    result = concatenate([i1, i3])

    assert expected_shape == result.shape

    # slice out z in addition to reduce the total feature number by 1/2
    i4 = intensities.where(np.logical_and(intensities.r == 0, intensities.z == 1), drop=True)
    expected_shape = (i1.shape[0] + i4.shape[0], 1, 3)
    result = concatenate([i1, i4])

    assert expected_shape == result.shape
