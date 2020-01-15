import copy

import numpy as np
import xarray as xr

from starfish.core.test.factories import SyntheticData
from starfish.core.types import Axes, Levels
from .factories import synthetic_stack
from ..imagestack import ImageStack


def divide(array, value):
    return array / value


def test_apply():
    """test that apply correctly applies a simple function across 2d tiles of a Stack"""
    stack = synthetic_stack()
    assert (stack.xarray == 1).all()
    output = stack.apply(divide, value=2)
    assert (output.xarray == 0.5).all()


def test_apply_positional():
    """test that apply correctly applies a simple function across 2d tiles of a Stack.  Unlike
    test_apply, the parameter is passed in as a positional parameter."""
    stack = synthetic_stack()
    assert (stack.xarray == 1).all()
    output = stack.apply(divide, 2, n_processes=1)
    assert (output.xarray == 0.5).all()


def test_apply_3d():
    """test that apply correctly applies a simple function across 3d volumes of a Stack"""
    stack = synthetic_stack()
    assert np.all(stack.xarray == 1)
    stack.apply(divide, in_place=True, value=4,
                group_by={Axes.ROUND, Axes.CH})
    assert (stack.xarray == 0.25).all()


def test_apply_labeled_dataset():
    """
    test that apply correctly applies a simple function across starfish-generated synthetic data
    """
    original = SyntheticData().spots()
    image = original.apply(divide, value=2)
    assert np.all(image.xarray == original.xarray / 2)


def test_apply_in_place():
    """
    test that apply correctly applies a simple function across a starfish stack without modifying
    original data
    """
    image = SyntheticData().spots()
    original = copy.deepcopy(image)
    image.apply(divide, value=2, in_place=True)
    assert np.all(image.xarray == original.xarray / 2)


def test_apply_single_process():
    """test that apply correctly applies a simple function across 2d tiles of a Stack"""
    stack = synthetic_stack()
    assert (stack.xarray == 1).all()
    output = stack.apply(divide, value=2, n_processes=1)
    assert (output.xarray == 0.5).all()

def test_apply_clipping_methods():
    """test that apply properly clips the imagestack"""

    # create a half-valued float array
    data = np.full((2, 2, 2, 5, 5), fill_value=0.5, dtype=np.float32)
    # set one value to max
    data[1, 1, 1, 1, 1] = 1

    imagestack = ImageStack.from_numpy(data)

    # max value after multiplication == 2, all other values == 1
    def apply_function(x):
        return x * 2

    # clip_method 0
    # all data are clipped to 1, setting all values to 1 (np.unique(pre_scaled) == [1, 2])
    res = imagestack.apply(apply_function, level_method=Levels.CLIP, in_place=False, n_processes=1)
    assert np.allclose(res.xarray.values, 1)

    # clip_method 1
    # all data are scaled, resulting in values being multiplied by 0.5, replicating the original
    # data
    res = imagestack.apply(
        apply_function, level_method=Levels.SCALE_SATURATED_BY_IMAGE, in_place=False, n_processes=1
    )
    assert np.allclose(imagestack.xarray, res.xarray)
    assert isinstance(imagestack.xarray, xr.DataArray)

    # clip_method 2
    res = imagestack.apply(
        apply_function, level_method=Levels.SCALE_SATURATED_BY_CHUNK, in_place=False, n_processes=1,
        group_by={Axes.CH, Axes.ROUND},
    )
    # any (round, ch) combination that was all 0.5 should now be all 1.
    assert np.allclose(res.sel({Axes.ROUND: 0, Axes.CH: 0}).xarray, 1)
    assert np.allclose(res.sel({Axes.ROUND: 1, Axes.CH: 0}).xarray, 1)
    assert np.allclose(res.sel({Axes.ROUND: 0, Axes.CH: 1}).xarray, 1)

    # the specific (round, ch) combination with the single "1" value should be scaled, and due to
    # construction, look like the original data.
    assert np.allclose(
        res.sel({Axes.ROUND: 1, Axes.CH: 1}).xarray,
        imagestack.sel({Axes.ROUND: 1, Axes.CH: 1}).xarray
    )
