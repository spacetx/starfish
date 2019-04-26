import numpy as np
import pytest
import xarray as xr

from starfish.core.image._filter.call_bases import CallBases
from starfish.types import Axes


def make_multicolor_image():
    '''
        Make a test tile
    '''

    image = np.ones((4, 2, 2), dtype='float32') * 0.25

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    for i in range(4):
        image[i, x[i], y[i]] = 1

    image[0, 0, 0] = 0.75
    image[2, 1, 1] = 0.9

    image_xr = xr.DataArray(image, dims=(Axes.CH.value, Axes.X.value, Axes.Y.value))

    return image_xr


def make_expected_base_calls(intensity_threshold=0, quality_threshold=0):
    '''
        Get the expected results depending on the test thresholds used.
    '''
    if intensity_threshold == 0 and quality_threshold == 0:
        base_calls = np.array(
            [[
                [0.86602545, 0], [0, 0]],
                [[0, 0.9176629], [0, 0]],
                [[0, 0], [0.9176629, 0]],
                [[0, 0], [0, 0.7188852]]], dtype='float32'
        )

    elif intensity_threshold == 0.8 and quality_threshold == 0:
        base_calls = np.array(
            [[
                [0, 0], [0, 0]],
                [[0, 0.9176629], [0, 0]],
                [[0, 0], [0.9176629, 0]],
                [[0, 0], [0, 0.7188852]]], dtype='float32'
        )

    elif intensity_threshold == 0 and quality_threshold == 0.75:
        base_calls = np.array(
            [[
                [0.86602545, 0], [0, 0]],
                [[0, 0.9176629], [0, 0]],
                [[0, 0], [0.9176629, 0]],
                [[0, 0], [0, 0]]], dtype='float32'
        )

    elif intensity_threshold == 0.8 and quality_threshold == 0.75:
        base_calls = np.array(
            [[
                [0, 0], [0, 0]],
                [[0, 0.9176629], [0, 0]],
                [[0, 0], [0.9176629, 0]],
                [[0, 0], [0, 0]]], dtype='float32'
        )

    bases_xr = xr.DataArray(base_calls, dims=(Axes.CH.value, Axes.X.value, Axes.Y.value))

    return bases_xr


@pytest.mark.parametrize("int_thresh, qual_thresh", [(0.8, 0), (0, 0.75), (0.8, 0.75)])
def test_call_bases(int_thresh: float, qual_thresh: float):

    # Instantiate the filter
    cb = CallBases()

    # Make the image and filter
    image_xr = make_multicolor_image()
    expected_result = make_expected_base_calls(
        intensity_threshold=int_thresh, quality_threshold=qual_thresh
    )
    base_image = cb._call_bases(
        image_xr, intensity_threshold=int_thresh, quality_threshold=qual_thresh
    )

    xr.testing.assert_equal(base_image, expected_result)
