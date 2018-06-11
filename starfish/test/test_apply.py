import numpy as np

from starfish.test.dataset_fixtures import synthetic_stack


def multiply(array, value):
    return array * value


def test_apply():
    """test that apply correctly applies a simple function across 2d tiles of a Stack"""
    stack = synthetic_stack()
    assert (stack.numpy_array == 1).all()
    stack.apply(multiply, value=2)
    assert (stack.numpy_array == 2).all()


def test_apply_3d():
    """test that apply correctly applies a simple function across 3d volumes of a Stack"""
    stack = synthetic_stack()
    assert np.all(stack.numpy_array == 1)
    stack.apply(multiply, value=4, is_volume=True)
    assert (stack.numpy_array == 4).all()
