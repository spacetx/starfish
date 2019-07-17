import numpy as np
import pytest
import xarray as xr

from starfish import ImageStack
from starfish.core.image.Filter.call_bases import CallBases
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


def make_image_stack():
    '''
        Make a test ImageStack

    '''

    # Make the test image
    test = np.ones((1, 4, 1, 2, 2), dtype='float32') * 0.5

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    for i in range(4):
        test[0, i, 0, x[i], y[i]] = 1
    test[0, 0, 0, 0, 0] = 0.75

    # Make the ImageStack
    test_stack = ImageStack.from_numpy(test)

    return test_stack


def make_expected_image_stack():
    '''
        Make the expected image stack result
    '''

    base_calls = np.array(
        [[[[[0, 0],
          [0, 0]]],
          [[[0, 0.755929],
           [0, 0]]],
          [[[0, 0],
           [0.755929, 0]]],
          [[[0, 0],
           [0, 0.755929]]]]], dtype='float32'
    )

    expected_stack = ImageStack.from_numpy(base_calls)

    return expected_stack


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


def test_image_stack_base_call():

    # Get the test stack and expected result
    test_stack = make_image_stack()
    expected_result = make_expected_image_stack()

    # Filter
    bc = CallBases(intensity_threshold=0.8)
    base_call = bc.run(test_stack, in_place=False)

    xr.testing.assert_equal(base_call.xarray, expected_result.xarray)
