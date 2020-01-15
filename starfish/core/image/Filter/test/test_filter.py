from typing import Tuple, Union

import numpy as np
import pytest
import xarray as xr

from starfish.core.image.Filter import element_wise_mult, gaussian_high_pass, mean_high_pass
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels, Number


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
        sigma=sigma, is_volume=is_volume, level_method=Levels.CLIP
    )
    result = ghp.run(image_stack, n_processes=1)
    assert np.sum(result.xarray) < sum_before  # type: ignore

@pytest.mark.parametrize('size, is_volume', [
    (1, False),
    ((1, 1), False),
    ((1, 0, 1), True)
])
def test_mean_high_pass(size: Union[Number, Tuple[Number]], is_volume: bool) -> None:
    """high pass is subtractive, sum of array should be less after running."""
    image_stack = random_data_image_stack_factory()
    sum_before = np.sum(image_stack.xarray)
    mhp = mean_high_pass.MeanHighPass(size=size, is_volume=is_volume, level_method=Levels.CLIP)
    result = mhp.run(image_stack)
    assert np.sum(result.xarray) < sum_before  # type: ignore


def test_element_wise_mult() -> None:
    image_stack = random_data_image_stack_factory()
    mult_array = xr.DataArray(
        np.array([[[[[0.5]]]]]),
        dims=('r', 'c', 'z', 'y', 'x')
    )
    ewm = element_wise_mult.ElementWiseMultiply(mult_array)
    multiplied = ewm.run(image_stack, in_place=False)
    assert isinstance(multiplied.xarray, xr.DataArray)  # type: ignore
    assert multiplied.xarray.equals(image_stack.xarray * .5)  # type: ignore
