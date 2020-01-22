import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pytest
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


@pytest.mark.parametrize(
    ["tile_format", "in_place"],
    [
        (ImageFormat.NUMPY, True),
        (ImageFormat.NUMPY, False),
        (ImageFormat.TIFF, True),
        (ImageFormat.TIFF, False),
    ]
)
def test_single_aligned_regular_fov(
        tmpdir,
        tile_format: ImageFormat,
        in_place: bool,
        rounds=(1, 2, 4),
        chs=(2, 3, 4),
        zplanes=(0, 1, 2),
        tile_height=100,
        tile_width=60,
        x_coords=(0.0, 0.1),
        y_coords=(0.1, 0.2),
        zplane_to_coords={0: 0.20, 1: 0.25, 2: 0.3},
        # default value is mutable, but it's for readability reasons!
):
    """Write the tiles for a single aligned (physical coordinates) regular (the dimensions have the
    same cardinality) FOV.  Then build an experiment from the tiles.  Finally, load the resulting
    experiment as an ImageStack and verify that the data matches."""
    tmpdir_path: Path = Path(tmpdir)
    tile_identifiers: Sequence[TileIdentifier] = [
        TileIdentifier(0, round_label, ch_label, zplane_label)
        for round_label in rounds
        for ch_label in chs
        for zplane_label in zplanes
    ]
    tile_fetcher: TileFetcher = tile_fetcher_factory(
        UniqueTiles,
        pass_tile_indices=True,
        fovs=[0],
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        tile_height=tile_height,
        tile_width=tile_width,
    )
    tile_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = {
        tile_identifier: {
            Coordinates.X: x_coords,
            Coordinates.Y: y_coords,
            Coordinates.Z: zplane_to_coords[tile_identifier.zplane_label],
        }
        for tile_identifier in tile_identifiers
    }

    write_tile_data(
        tmpdir_path,
        FieldOfView.PRIMARY_IMAGES,
        tile_format,
        tile_identifiers,
        tile_fetcher)

    coordinates_csv_path = tmpdir_path / "coordinates.csv"
    rows = render_coordinates_to_rows(tile_coordinates)
    write_coordinates_csv(coordinates_csv_path, rows, True)

    # Sleeping by 1 second will result in different timestamps written to TIFF files (the timestamp
    # in the TIFF header has 1 second resolution).  This exposes potential bugs, depending on the
    # nature of the bug and whether in_place is True.
    if tile_format == ImageFormat.TIFF:
        time.sleep(1)

    format_structured_dataset(
        os.fspath(tmpdir_path),
        os.fspath(coordinates_csv_path),
        os.fspath(tmpdir_path),
        tile_format,
        in_place,
    )

    # load the data and verify it.
    exp = Experiment.from_json(os.fspath(tmpdir_path / "experiment.json"))
    fov = exp.fov()
    stack = fov.get_image(FieldOfView.PRIMARY_IMAGES)

    for round_label in rounds:
        for ch_label in chs:
            for zplane_label in zplanes:
                data, _ = stack.get_slice({
                    Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label
                })
                expected_data = unique_data(
                    0, rounds.index(round_label), chs.index(ch_label), zplanes.index(zplane_label),
                    1, len(rounds), len(chs), len(zplanes),
                    tile_height, tile_width,
                )
                assert np.allclose(data, expected_data)

    for selectors in stack._iter_axes({Axes.ZPLANE}):
        zplane_label = selectors[Axes.ZPLANE]
        verify_physical_coordinates(
            stack, x_coords, y_coords, zplane_to_coords[zplane_label], selectors[Axes.ZPLANE])


def test_single_ragged_fov(
        tmpdir,
        raw_tile_identifiers=(
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (1, 0, 0),
        ),
        tile_height=100,
        tile_width=60,
        x_coords=(0.0, 0.1),
        y_coords=(0.1, 0.2),
        zplane_to_coords={0: 0.20, 1: 0.25, 2: 0.3},
        # default value is mutable, but it's for readability reasons!
):
    """Write the tiles for a single aligned (physical coordinates) ragged (the dimensions do not
    always have the same cardinality) FOV.  Then build an experiment from the tiles.  Finally, load
    the resulting experiment as an ImageStack and verify that the data matches."""
    tmpdir_path: Path = Path(tmpdir)
    tile_identifiers: Sequence[TileIdentifier] = [
        TileIdentifier(0, round_label, ch_label, zplane_label)
        for round_label, ch_label, zplane_label in raw_tile_identifiers
    ]
    rounds = sorted(set([round_label for round_label, _, _ in raw_tile_identifiers]))
    chs = sorted(set([ch_label for _, ch_label, _ in raw_tile_identifiers]))
    zplanes = sorted(set([zplane_label for _, _, zplane_label in raw_tile_identifiers]))
    tile_fetcher: TileFetcher = tile_fetcher_factory(
        UniqueTiles,
        pass_tile_indices=True,
        fovs=[0],
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        tile_height=tile_height,
        tile_width=tile_width,
    )
    tile_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = {
        tile_identifier: {
            Coordinates.X: x_coords,
            Coordinates.Y: y_coords,
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
    fov = exp.fov()

    for loaded_rounds, expected_chs, expected_zplanes in (
            ({0}, {0, 1}, {0, 1, 2}),
            ({1}, {0}, {0}),
    ):
        stacks = list(fov.get_images(FieldOfView.PRIMARY_IMAGES, rounds=loaded_rounds))

        assert len(stacks) == 1
        stack = stacks[0]
        assert set(stack.axis_labels(Axes.CH)) == set(expected_chs)
        assert set(stack.axis_labels(Axes.ZPLANE)) == set(expected_zplanes)

        for round_label in loaded_rounds:
            for ch_label in expected_chs:
                for zplane_label in expected_zplanes:
                    data, _ = stack.get_slice({
                        Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label
                    })
                    expected_data = unique_data(
                        0,
                        rounds.index(round_label), chs.index(ch_label), zplanes.index(zplane_label),
                        1, len(rounds), len(chs), len(zplanes),
                        tile_height, tile_width,
                    )
                    assert np.allclose(data, expected_data)

        for selectors in stack._iter_axes({Axes.ZPLANE}):
            zplane_label = selectors[Axes.ZPLANE]
            verify_physical_coordinates(
                stack, x_coords, y_coords, zplane_to_coords[zplane_label], selectors[Axes.ZPLANE])
