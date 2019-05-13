import numpy as np
import pytest
import xarray as xr

from starfish import ImageStack
from starfish.core.image._filter.reduce import Reduce
from starfish.types import Axes


def make_image_stack():
    '''
        Make a test ImageStack

    '''

    # Make the test image
    test = np.ones((2, 4, 1, 2, 2), dtype='float32') * 0.1

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    for i in range(4):
        test[0, i, 0, x[i], y[i]] = 1
    test[0, 0, 0, 0, 0] = 0.75

    # Make the ImageStack
    test_stack = ImageStack.from_numpy(test)

    return test_stack


def make_expected_image_stack(func):
    '''
        Make the expected image stack result
    '''

    if func == 'max':
        reduced = np.array(
            [[[[[0.75, 0.1],
                [0.1, 0.1]]],
                [[[0.1, 1],
                  [0.1, 0.1]]],
                [[[0.1, 0.1],
                  [1, 0.1]]],
                [[[0.1, 0.1],
                  [0.1, 1]]]]], dtype='float32'
        )
    elif func == 'mean':
        reduced = np.array(
            [[[[[0.425, 0.1],
              [0.1, 0.1]]],
              [[[0.1, 0.55],
                [0.1, 0.1]]],
              [[[0.1, 0.1],
                [0.55, 0.1]]],
              [[[0.1, 0.1],
                [0.1, 0.55]]]]], dtype='float32'
        )
    elif func == 'sum':
        reduced = np.array(
            [[[[[0.85, 0.2],
              [0.2, 0.2]]],
              [[[0.2, 1],
                [0.2, 0.2]]],
              [[[0.2, 0.2],
                [1, 0.2]]],
              [[[0.2, 0.2],
                [0.2, 1]]]]], dtype='float32'
        )

    expected_stack = ImageStack.from_numpy(reduced)

    return expected_stack


@pytest.mark.parametrize("func", ['max', 'mean', 'sum'])
def test_image_stack_reduce(func):

    # Get the test stack and expected result
    test_stack = make_image_stack()
    expected_result = make_expected_image_stack(func=func)

    # Filter
    red = Reduce(dims=[Axes.ROUND], func=func)
    reduced = red.run(test_stack, in_place=False)

    xr.testing.assert_equal(reduced.xarray, expected_result.xarray)
