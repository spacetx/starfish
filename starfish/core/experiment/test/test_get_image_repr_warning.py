"""
Tests for ImageStack repr with aligned groups and FieldOfView.get_image warning.
"""

import numpy as np
import pytest
from slicedimage import TileSet

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.types import Axes
from ..experiment import FieldOfView


def test_imagestack_repr_shows_aligned_group():
    """Test that ImageStack repr shows aligned group info when attributes are set."""
    # Create a small ImageStack from numpy array
    array = np.zeros((1, 1, 1, 10, 10), dtype=np.float32)
    stack = ImageStack.from_numpy(array)

    # Set aligned group attributes
    stack.aligned_group = 1
    stack._coordinate_group_count = 3

    # Check repr contains aligned group info (1-based: group 1 -> 2/3)
    repr_str = repr(stack)
    assert "(aligned_group=2/3)" in repr_str

    # Check _repr_html_ returns pre-formatted HTML with same info
    html = stack._repr_html_()
    assert html.startswith("<pre>")
    assert html.endswith("</pre>")
    assert "(aligned_group=2/3)" in html


def test_get_image_warns_and_sets_attrs(monkeypatch):
    """Test that get_image warns when multiple aligned groups exist and sets attributes."""

    # Create two dummy CropParameters to simulate two aligned groups
    crop_param_1 = CropParameters(
        permitted_rounds=[0],
        permitted_chs=[0],
        permitted_zplanes=[0]
    )
    crop_param_2 = CropParameters(
        permitted_rounds=[1],
        permitted_chs=[0],
        permitted_zplanes=[0]
    )

    # Patch parse_aligned_groups to return two groups
    def mock_parse_aligned_groups(tileset, rounds=None, chs=None, zplanes=None, x=None, y=None):
        return [crop_param_1, crop_param_2]

    monkeypatch.setattr(
        CropParameters,
        'parse_aligned_groups',
        mock_parse_aligned_groups
    )

    # Create a mock ImageStack to return from from_tileset
    mock_stack = ImageStack.from_numpy(np.zeros((1, 1, 1, 10, 10), dtype=np.float32))

    # Patch ImageStack.from_tileset to return our mock stack
    def mock_from_tileset(tileset, crop_parameters=None):
        return mock_stack

    monkeypatch.setattr(
        ImageStack,
        'from_tileset',
        mock_from_tileset
    )

    # Create a dummy TileSet
    dummy_tileset = TileSet(
        [Axes.X, Axes.Y, Axes.CH, Axes.ZPLANE, Axes.ROUND],
        {Axes.CH: 1, Axes.ROUND: 2, Axes.ZPLANE: 1},
        {Axes.Y: 10, Axes.X: 10}
    )

    # Create a FieldOfView with the dummy tileset
    fov = FieldOfView("test_fov", {'primary': dummy_tileset})

    # Assert that a warning is raised when calling get_image
    with pytest.warns(UserWarning, match="Multiple aligned coordinate groups detected"):
        stack = fov.get_image('primary')

    # Verify the returned stack has the correct attributes
    assert hasattr(stack, 'aligned_group')
    assert hasattr(stack, '_coordinate_group_count')
    assert stack.aligned_group == 0
    assert stack._coordinate_group_count == 2
