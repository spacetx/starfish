import warnings
from collections import OrderedDict

import numpy as np
import pytest

from starfish import data
from starfish import ImageStack
from starfish.core.image.Filter.reduce import Reduce
from starfish.core.imagestack.test.factories import imagestack_with_coords_factory
from starfish.core.imagestack.test.imagestack_test_utils import verify_physical_coordinates
from starfish.types import Axes, FunctionSource, PhysicalCoordinateTypes


X_COORDS = 1, 2
Y_COORDS = 4, 6
Z_COORDS = 1, 3


def make_image_stack():
    """Make a test ImageStack."""

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
    """Make the expected image stack result"""

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
    elif func == 'norm':
        reduced = np.array(
            [[[[[0.75663730, 0.14142136],
                [0.14142136, 0.14142136]]],
              [[[0.14142136, 1.00000000],
                [0.14142136, 0.14142136]]],
              [[[0.14142136, 0.14142136],
                [1.00000000, 0.14142136]]],
              [[[0.14142136, 0.14142136],
                [0.14142136, 1.00000000]]]]],
            dtype=np.float32
        )
    else:
        raise ValueError("Unsupported func")

    expected_stack = ImageStack.from_numpy(reduced)

    return expected_stack


@pytest.mark.parametrize(
    "expected_result,func,module,kwargs",
    [
        (make_expected_image_stack('max'), 'max', None, None),
        (make_expected_image_stack('mean'), 'mean', None, None),
        (make_expected_image_stack('sum'), 'sum', None, None),
        (
            make_expected_image_stack('norm'),
            'linalg.norm',
            FunctionSource.scipy,
            {'ord': 2},
        ),
        (
            make_expected_image_stack('norm'),
            FunctionSource.scipy('linalg.norm'),
            None,
            {'ord': 2},
        ),
    ]
)
def test_image_stack_reduce(expected_result, func, module, kwargs):

    # Get the test stack and expected result
    test_stack = make_image_stack()

    # Filter
    if kwargs is None:
        actual_kwargs = {}
    else:
        actual_kwargs = kwargs.copy()
    if module is not None:
        actual_kwargs['module'] = module
    red = Reduce(dims=[Axes.ROUND], func=func, **actual_kwargs)
    reduced = red.run(test_stack)

    assert np.allclose(reduced.xarray, expected_result.xarray)


def test_image_stack_module_deprecated():
    """Specifying the function as a string and passing in a module should generate a warning."""
    with warnings.catch_warnings(record=True) as all_warnings:
        Reduce(dims=[Axes.ROUND], func="max", module=FunctionSource.np)

        assert DeprecationWarning in (warning.category for warning in all_warnings)


def test_image_stack_module_with_functionsourcebundle():
    """Specifying the function as a FunctionSourceBundle and passing in a module should raise an
    Exception."""
    with pytest.raises(ValueError):
        Reduce(dims=[Axes.ROUND], func=FunctionSource.np("max"), module=FunctionSource.np)


def test_max_projection_preserves_coordinates():
    e = data.ISS(use_test_data=True)
    nuclei = e.fov().get_image('nuclei')

    red = Reduce(dims=[Axes.ROUND, Axes.CH, Axes.ZPLANE], func='max')
    nuclei_proj = red.run(nuclei)

    # Since this data already has only 1 round, 1 ch, 1 zplane
    # let's just assert that the max_proj operation didn't change anything
    assert nuclei.xarray.equals(nuclei_proj.xarray)

    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 3), (Axes.Y, 10), (Axes.X, 10)])

    # Create stack with coordinates, verify coords unaffected by max_poj
    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)

    stack_proj = red.run(stack)
    expected_z = np.average(Z_COORDS)
    verify_physical_coordinates(stack_proj, X_COORDS, Y_COORDS, expected_z)
