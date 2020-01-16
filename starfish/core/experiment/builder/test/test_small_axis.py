import warnings
from typing import Callable, Collection, Mapping, Sequence

import pytest

from starfish.core.errors import DataFormatWarning
from starfish.core.types import Coordinates, CoordinateValue
from .factories.all_purpose import collection_factory
from .factories.unique_tiles import UniqueTiles
from ..builder import TileIdentifier


@pytest.mark.parametrize(
    "num_round_labels, num_ch_labels, num_zplane_labels, tile_height, tile_width, has_warnings",
    (
        (1, 1, 1, 100, 100, False),
        (1, 1, 1000, 100, 100, True),
        (1, 1000, 1, 100, 100, True),
        (1000, 1, 1, 100, 100, True),
        (1000, 1, 1, 10000, 100, True),
        (1000, 1, 1, 100, 10000, True),
    )
)
def test_small_axis(
        num_round_labels,
        num_ch_labels,
        num_zplane_labels,
        tile_height,
        tile_width,
        has_warnings,
):
    tile_identifiers: Collection[TileIdentifier] = [
        TileIdentifier(0, round_label, ch_label, zplane_label)
        for round_label in range(num_round_labels)
        for ch_label in range(num_ch_labels)
        for zplane_label in range(num_zplane_labels)
    ]

    def make_tile_coordinate_callback(
            all_zplane_labels: Sequence[int]
    ) -> Callable[[TileIdentifier], Mapping[Coordinates, CoordinateValue]]:
        def tile_coordinate_callback(
                tile_identifier: TileIdentifier
        ) -> Mapping[Coordinates, CoordinateValue]:
            zplane_offset = all_zplane_labels.index(tile_identifier.zplane_label)
            return {
                Coordinates.X: (0.0, 0.1),
                Coordinates.Y: (0.0, 0.1),
                Coordinates.Z: zplane_offset * 0.1,
            }

        return tile_coordinate_callback

    with warnings.catch_warnings(record=True) as warnings_:
        collection_factory(
            UniqueTiles,
            tile_identifiers,
            make_tile_coordinate_callback(
                sorted(set(tile_identifier.zplane_label for tile_identifier in tile_identifiers))),
            tile_height,
            tile_width,
        )
        assert has_warnings == (
            0 != len(
                [warning for warning in warnings_
                 if issubclass(warning.category, DataFormatWarning)]))
