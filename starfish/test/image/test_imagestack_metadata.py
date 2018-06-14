from typing import Any

import pytest

from starfish.constants import Indices
from starfish.util.synthesize import DEFAULT_NUM_HYB, DEFAULT_NUM_CH, DEFAULT_NUM_Z
from starfish.image import ImageStack


def test_metadata():
    """
    Normal situation where all the tiles have uniform keys for both indices and extras.
    """
    def tile_extras_provider(hyb: int, ch: int, z: int) -> Any:
        return {
            'random_key': {
                Indices.HYB: hyb,
                Indices.CH: ch,
                Indices.Z: z,
            }
        }

    stack = ImageStack.synthetic_stack(
        tile_extras_provider=tile_extras_provider,
    )
    table = stack.tile_metadata
    assert len(table) == DEFAULT_NUM_HYB * DEFAULT_NUM_CH * DEFAULT_NUM_Z


def test_missing_extras():
    """
    If the extras are not present on some of the tiles, it should still work.
    """
    def tile_extras_provider(hyb: int, ch: int, z: int) -> Any:
        if hyb == 0:
            return {
                'random_key': {
                    Indices.HYB: hyb,
                    Indices.CH: ch,
                    Indices.Z: z,
                }
            }
        else:
            return None

    stack = ImageStack.synthetic_stack(
        tile_extras_provider=tile_extras_provider,
    )
    table = stack.tile_metadata
    assert len(table) == DEFAULT_NUM_HYB * DEFAULT_NUM_CH * DEFAULT_NUM_Z


def test_conflict():
    """
    Tiles that have extras that conflict with indices should produce an error.
    """
    def tile_extras_provider(hyb: int, ch: int, z: int) -> Any:
        return {
            Indices.HYB: hyb,
            Indices.CH: ch,
            Indices.Z: z,
        }

    stack = ImageStack.synthetic_stack(
        tile_extras_provider=tile_extras_provider,
    )
    with pytest.raises(ValueError):
        stack.tile_metadata
