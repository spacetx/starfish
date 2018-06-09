from copy import deepcopy

import numpy as np

from starfish.test.dataset_fixtures import synthetic_stack_factory


def multiply(array, value):
    return array * value


def test_apply(synthetic_stack_factory):
    """test that apply correctly applies a simple function across 2d tiles of a Stack

    Parameters
    ----------
    synthetic_stack : Stack
        synthetic data pytest fixture that passes a Stack object

    """
    synthetic_stack = deepcopy(synthetic_stack_factory())
    assert (synthetic_stack.numpy_array == 1).all()
    synthetic_stack.apply(multiply, value=2)
    assert (synthetic_stack.numpy_array == 2).all()


def test_apply_3d(synthetic_stack_factory):
    """test that apply correctly applies a simple function across 3d volumes of a Stack

    Parameters
    ----------
    synthetic_stack : Stack
        synthetic data pytest fixture that passes a Stack object

    """
    synthetic_stack = deepcopy(synthetic_stack_factory())
    assert np.all(synthetic_stack.numpy_array == 1)
    synthetic_stack.apply(multiply, value=4, is_volume=True)
    assert (synthetic_stack.numpy_array == 4).all()
