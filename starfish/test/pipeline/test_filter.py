from typing import Tuple, Union

import numpy as np
import pytest

from starfish.experiment import Experiment
from starfish.image._filter import gaussian_high_pass
from starfish.test.dataset_fixtures import merfish_stack
from starfish.types import Number


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass(merfish_stack: Experiment, sigma: Union[Number, Tuple[Number]]):
    """high pass is subtractive, sum of array should be less after running

    Parameters
    ----------
    merfish_stack : Experiment
        pytest fixture that exposes a starfish.io.Stack object containing MERFISH testing data
    sigma : Union[int, Tuple[int]]
        the standard deviation of the gaussian kernel
    """
    image = merfish_stack.fov().primary_image
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    sum_before = np.sum(image.numpy_array)
    ghp.run(image)
    assert np.sum(image.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 1)))
def test_gaussian_high_pass_apply(merfish_stack: Experiment, sigma: Union[Number, Tuple[Number]]):
    """same as test_gaussian_high_pass, but tests apply loop functionality"""
    image = merfish_stack.fov().primary_image
    sum_before = np.sum(image.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma)
    ghp.run(image)
    assert np.sum(image.numpy_array) < sum_before


@pytest.mark.parametrize('sigma', (1, (1, 0, 1)))
def test_gaussian_high_pass_apply_3d(
        merfish_stack: Experiment, sigma: Union[Number, Tuple[Number]]
):
    """same as test_gaussian_high_pass, but tests apply loop functionality"""
    image = merfish_stack.fov().primary_image
    sum_before = np.sum(image.numpy_array)
    ghp = gaussian_high_pass.GaussianHighPass(sigma=sigma, is_volume=True)
    ghp.run(image)
    assert np.sum(image.numpy_array) < sum_before
