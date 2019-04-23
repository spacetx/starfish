from typing import Tuple, Union

import numpy as np
import pytest

from starfish.core.image._filter import gaussian_high_pass, mean_high_pass
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Clip, Number


def random_data_image_stack_factory():
    data = np.random.uniform(0, 1, 100).reshape(1, 1, 1, 10, 10).astype(np.float32)
    return ImageStack.from_numpy(data)


@pytest.mark.parametrize('sigma, is_volume', [
    (1, False),
    ((1, 1), False),
    ((1, 0, 1), True)
])
def test_gaussian_high_pass(sigma: Union[Number, Tuple[Number]], is_volume: bool) -> None:
    """high pass is subtractive, sum of array should be less after running."""
    image_stack = random_data_image_stack_factory()
    sum_before = np.sum(image_stack.xarray)
    ghp = gaussian_high_pass.GaussianHighPass(
        sigma=sigma, is_volume=is_volume, clip_method=Clip.CLIP
    )
    result = ghp.run(image_stack, n_processes=1)
    assert np.sum(result.xarray) < sum_before

@pytest.mark.parametrize('size, is_volume', [
    (1, False),
    ((1, 1), False),
    ((1, 0, 1), True)
])
def test_mean_high_pass(size: Union[Number, Tuple[Number]], is_volume: bool) -> None:
    """high pass is subtractive, sum of array should be less after running."""
    image_stack = random_data_image_stack_factory()
    sum_before = np.sum(image_stack.xarray)
    mhp = mean_high_pass.MeanHighPass(size=size, is_volume=is_volume, clip_method=Clip.CLIP)
    result = mhp.run(image_stack)
    assert np.sum(result.xarray) < sum_before
