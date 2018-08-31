from typing import Tuple, Union

import numpy as np
import pytest

from starfish.image._filter import gaussian_high_pass
from starfish.stack import ImageStack
from starfish.types import Number


def random_data_image_stack_factory():
    data = np.random.uniform(0, 10, 100).reshape(1, 1, 1, 10, 10).astype(np.uint16)
    return ImageStack.from_numpy_array(data)


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass(sigma: Union[Number, Tuple[Number]]) -> None:
    """high pass is subtractive, sum of array should be less after running."""
    image_stack = random_data_image_stack_factory()
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    sum_before = np.sum(image_stack.numpy_array)
    ghp.run(image_stack)
    assert np.sum(image_stack.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass_apply(sigma: Union[Number, Tuple[Number]]) -> None:
    """same as test_gaussian_high_pass, but tests apply loop functionality."""
    image_stack = random_data_image_stack_factory()
    sum_before = np.sum(image_stack.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    ghp.run(image_stack)
    assert np.sum(image_stack.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 0, 1)))
def test_gaussian_high_pass_apply_3d(sigma: Union[Number, Tuple[Number]]) -> None:
    """same as test_gaussian_high_pass, but tests apply loop functionality in 3d"""
    image_stack = random_data_image_stack_factory()
    sum_before = np.sum(image_stack.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma, is_volume=True)
    ghp.run(image_stack)
    assert np.sum(image_stack.numpy_array) < sum_before
