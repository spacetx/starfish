import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from slicedimage import ImageFormat

from starfish.core.experiment.experiment import Experiment, FieldOfView
from starfish.core.imagestack.test.imagestack_test_utils import verify_physical_coordinates
from starfish.core.types import Axes, Coordinates, CoordinateValue
from .util import (
    render_coordinates_to_rows,
    write_coordinates_csv,
    write_tile_data,
)
from ..factories import unique_data
from ..factories.unique_tiles import UniqueTiles
from ...builder import (
    tile_fetcher_factory,
    TileFetcher,
    TileIdentifier,
)
from ...structured_formatter import format_structured_dataset


def test_multiple_aligned_regular_fov(
        tmpdir,
        fovs=(0, 1, 2, 3),
        rounds=(1, 2, 4),
        chs=(2, 3, 4),
        zplanes=(0, 1, 2),
        tile_height=100,
        tile_width=60,
        fov_to_x_coords={
            0: (0.0, 0.1),
            1: (0.0, 0.1),
            2: (0.1, 0.2),
            3: (0.1, 0.2),
        },
        fov_to_y_coords={
            0: (0.0, 0.1),
            1: (0.1, 0.2),
            2: (0.0, 0.1),
            3: (0.1, 0.2),
        },
        zplane_to_coords={0: 0.20, 1: 0.25, 2: 0.3},
        # default value is mutable, but it's for readability reasons!
):
    """Write the tiles for a multi-fov aligned (physical coordinates) regular (the dimensions have
    the same cardinality) image.  Then build an experiment from the tiles.  Finally, load the
    resulting experiment as an ImageStack and verify that the data matches."""
    tmpdir_path: Path = Path(tmpdir)
    tile_identifiers: Sequence[TileIdentifier] = [
        TileIdentifier(fov_id, round_label, ch_label, zplane_label)
        for fov_id in fovs
        for round_label in rounds
        for ch_label in chs
        for zplane_label in zplanes
    ]
    tile_fetcher: TileFetcher = tile_fetcher_factory(
        UniqueTiles,
        pass_tile_indices=True,
        fovs=fovs,
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        tile_height=tile_height,
        tile_width=tile_width,
    )
    tile_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = {
        tile_identifier: {
            Coordinates.X: fov_to_x_coords[tile_identifier.fov_id],
            Coordinates.Y: fov_to_y_coords[tile_identifier.fov_id],
            Coordinates.Z: zplane_to_coords[tile_identifier.zplane_label],
        }
        for tile_identifier in tile_identifiers
    }

    write_tile_data(
        tmpdir_path,
        FieldOfView.PRIMARY_IMAGES,
        ImageFormat.TIFF,
        tile_identifiers,
        tile_fetcher)

    coordinates_csv_path = tmpdir_path / "coordinates.csv"
    rows = render_coordinates_to_rows(tile_coordinates)
    write_coordinates_csv(coordinates_csv_path, rows, True)

    format_structured_dataset(
        os.fspath(tmpdir_path),
        os.fspath(coordinates_csv_path),
        os.fspath(tmpdir_path),
        ImageFormat.TIFF,
        False,
    )

    # load the data and verify it.
    exp = Experiment.from_json(os.fspath(tmpdir_path / "experiment.json"))

    for fov_id in fovs:
        fov = exp.fov(lambda fieldofview: fieldofview.name == f"fov_{fov_id:03}")
        stack = fov.get_image(FieldOfView.PRIMARY_IMAGES)
        for round_label in rounds:
            for ch_label in chs:
                for zplane_label in zplanes:
                    data, _ = stack.get_slice({
                        Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label
                    })
                    expected_data = unique_data(
                        fovs.index(fov_id),
                        rounds.index(round_label),
                        chs.index(ch_label),
                        zplanes.index(zplane_label),
                        len(fovs), len(rounds), len(chs), len(zplanes),
                        tile_height, tile_width,
                    )
                    assert np.allclose(data, expected_data)

        for selectors in stack._iter_axes({Axes.ZPLANE}):
            zplane_label = selectors[Axes.ZPLANE]
            verify_physical_coordinates(
                stack,
                fov_to_x_coords[fov_id],
                fov_to_y_coords[fov_id],
                zplane_to_coords[zplane_label],
                selectors[Axes.ZPLANE])
