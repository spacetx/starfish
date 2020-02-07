from typing import Mapping, Union

import numpy as np
import pytest

from starfish.core.experiment.builder.builder import tile_fetcher_factory
from starfish.core.experiment.builder.test.factories.unique_tiles import unique_data, UniqueTiles
from starfish.core.types import Axes, Coordinates, CoordinateValue
from ..imagestack import ImageStack


class UniqueTilesWithCoordinates(UniqueTiles):
    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }


@pytest.mark.parametrize(
    "group_by",
    [
        None,
        set((Axes.ROUND,)),
        set((Axes.ROUND, Axes.CH)),
        set((Axes.ROUND, Axes.CH, Axes.ZPLANE)),
        set((Axes.ZPLANE,)),
    ]
)
def test_from_tilefetcher(
        group_by,
        rounds=(0, 1, 2, 3),
        chs=(0, 1, 3),
        zplanes=(0, 1),
        tile_height=400,
        tile_width=500,
):
    tile_fetcher = tile_fetcher_factory(
        UniqueTilesWithCoordinates,
        pass_tile_indices=True,
        fovs=[0],
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        tile_height=tile_height,
        tile_width=tile_width,
    )
    stack = ImageStack.from_tilefetcher(
        tile_fetcher,
        tile_shape={Axes.Y: tile_height, Axes.X: tile_width},
        fov=0,
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        group_by=group_by,
    )

    assert stack.shape == {
        Axes.ROUND: len(rounds),
        Axes.CH: len(chs),
        Axes.ZPLANE: len(zplanes),
        Axes.Y.value: tile_height,
        Axes.X.value: tile_width,
    }

    for round_label in rounds:
        for ch_label in chs:
            for zplane_label in zplanes:
                expected_data = unique_data(
                    0,
                    rounds.index(round_label),
                    chs.index(ch_label),
                    zplanes.index(zplane_label),
                    1,
                    len(rounds),
                    len(chs),
                    len(zplanes),
                    tile_height,
                    tile_width,
                )
                actual_data = stack.get_slice(
                    {Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label})[0]

                assert np.all(expected_data == actual_data)
