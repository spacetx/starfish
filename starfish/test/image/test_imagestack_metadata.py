from typing import Any

import pytest

from starfish.constants import Indices
from starfish.image import ImageStack

NUM_ROUND = 4
NUM_CH = 2
NUM_Z = 12


def test_metadata():
    """
    Normal situation where all the tiles have uniform keys for both indices and extras.
    """
    def tile_extras_provider(round_: int, ch: int, z: int) -> Any:
        return {
            'random_key': {
                Indices.ROUND: round_,
                Indices.CH: ch,
                Indices.Z: z,
            }
        }

    stack = ImageStack.synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_Z, tile_extras_provider=tile_extras_provider,
    )
    table = stack.tile_metadata
    assert len(table) == NUM_ROUND * NUM_CH * NUM_Z


def test_missing_extras():
    """
    If the extras are not present on some of the tiles, it should still work.
    """
    def tile_extras_provider(round_: int, ch: int, z: int) -> Any:
        if round_ == 0:
            return {
                'random_key': {
                    Indices.ROUND: round_,
                    Indices.CH: ch,
                    Indices.Z: z,
                }
            }
        else:
            return None

    stack = ImageStack.synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_Z, tile_extras_provider=tile_extras_provider,
    )
    table = stack.tile_metadata
    assert len(table) == NUM_ROUND * NUM_CH * NUM_Z


def test_conflict():
    """
    Tiles that have extras that conflict with indices should produce an error.
    """
    def tile_extras_provider(round_: int, ch: int, z: int) -> Any:
        return {
            Indices.ROUND: round_,
            Indices.CH: ch,
            Indices.Z: z,
        }

    stack = ImageStack.synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_Z, tile_extras_provider=tile_extras_provider,
    )
    with pytest.raises(ValueError):
        stack.tile_metadata
