import pytest

from starfish.core.experiment.builder.defaultproviders import OnesTile, tile_fetcher_factory
from starfish.core.types import Axes
from .factories import synthetic_stack

NUM_ROUND = 4
NUM_CH = 2
NUM_ZPLANE = 12


class OnesTilesWithExtras(OnesTile):
    def __init__(self, extras: dict, *args, **kwargs) -> None:
        super().__init__({Axes.Y: 10, Axes.X: 10})
        self._extras = extras

    @property
    def extras(self):
        return self._extras


def test_metadata():
    """
    Normal situation where all the tiles have uniform keys for both indices and extras.
    """
    tile_fetcher = tile_fetcher_factory(
        OnesTilesWithExtras,
        False,
        {
            'random_key': {
                'hello': "world",
            }
        }
    )

    stack = synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_ZPLANE, tile_fetcher=tile_fetcher,
    )
    table = stack.tile_metadata
    assert len(table) == NUM_ROUND * NUM_CH * NUM_ZPLANE


def test_missing_extras():
    """
    If the extras are not present on some of the tiles, it should still work.
    """
    class OnesTilesWithExtrasMostly(OnesTile):
        def __init__(self, fov, r, ch, z, extras: dict) -> None:
            super().__init__({Axes.Y: 10, Axes.X: 10})
            self.fov = fov
            self._extras = extras

        @property
        def extras(self):
            if self.fov == 0:
                return None
            return self._extras

    tile_fetcher = tile_fetcher_factory(
        OnesTilesWithExtrasMostly,
        True,
        {
            'random_key': {
                'hello': "world",
            }
        }
    )

    stack = synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_ZPLANE, tile_fetcher=tile_fetcher,
    )
    table = stack.tile_metadata
    assert len(table) == NUM_ROUND * NUM_CH * NUM_ZPLANE


def test_conflict():
    """
    Tiles that have extras that conflict with indices should produce an error.
    """
    tile_fetcher = tile_fetcher_factory(
        OnesTilesWithExtras,
        False,
        {
            Axes.ROUND: {
                'hello': "world",
            }
        }
    )

    stack = synthetic_stack(
        num_round=NUM_ROUND, num_ch=NUM_CH, num_z=NUM_ZPLANE, tile_fetcher=tile_fetcher,
    )
    with pytest.raises(ValueError):
        stack.tile_metadata
