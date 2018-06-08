from numbers import Number
from typing import Tuple, Union

import numpy as np
import pytest

from starfish.io import Stack
from starfish.test.dataset_fixtures import merfish_stack
from starfish.pipeline.filter import gaussian_high_pass


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass(merfish_stack: Stack, sigma: Union[Number, Tuple[Number]]):
    """high pass is subtractive, sum of array should be less after running

    Parameters
    ----------
    merfish_stack : Stack
        pytest fixture that exposes a starfish.io.Stack object containing MERFISH testing data
    sigma : Union[int, Tuple[int]]
        the standard deviation of the gaussian kernel
    """
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    sum_before = np.sum(merfish_stack.image.numpy_array)
    ghp.filter(merfish_stack.image)
    assert np.sum(merfish_stack.image.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass_apply(merfish_stack: Stack, sigma: Union[Number, Tuple[Number]]):
    """same as test_gaussian_high_pass, but tests apply loop functionality"""
    sum_before = np.sum(merfish_stack.image.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    ghp.filter(merfish_stack.image)
    assert np.sum(merfish_stack.image.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 0, 1)))
def test_gaussian_high_pass_apply_3d(merfish_stack: Stack, sigma: Union[Number, Tuple[Number]]):
    """same as test_gaussian_high_pass, but tests apply loop functionality"""
    sum_before = np.sum(merfish_stack.image.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma, is_volume=True)
    ghp.filter(merfish_stack.image)
    assert np.sum(merfish_stack.image.numpy_array) < sum_before
