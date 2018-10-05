import copy

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Indices
from starfish.util.synthesize import SyntheticData


def divide(array, value):
    return array / value


def test_apply():
    """test that apply correctly applies a simple function across 2d tiles of a Stack"""
    stack = ImageStack.synthetic_stack()
    assert (stack.xarray == 1).all()
    output = stack.apply(divide, value=2)
    assert (output.xarray == 0.5).all()


def test_apply_3d():
    """test that apply correctly applies a simple function across 3d volumes of a Stack"""
    stack = ImageStack.synthetic_stack()
    assert np.all(stack.xarray == 1)
    stack.apply(divide, in_place=True, value=4,
                split_by={Indices.Z.value, Indices.Y.value, Indices.X.value})
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
