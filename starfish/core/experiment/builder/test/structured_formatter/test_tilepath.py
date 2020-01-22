"""These are tests that place the tiles in different locations that are still considered valid.
For instance, primary-f0_r000000_c0_z0.tiff should still resolve to tile in round 0, ch 0, zplane 0
in FOV 0.  We should also be able to put tiles in arbitrary subdirectories and still have everything
work.
"""

import os
import random
import string
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple

import numpy as np
from slicedimage import ImageFormat

from starfish.core.experiment.experiment import Experiment, FieldOfView
from starfish.core.imagestack.test.imagestack_test_utils import verify_physical_coordinates
from starfish.core.types import Axes, Coordinates, CoordinateValue, Number
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


def tilepath_test(
        tmpdir_path: Path,
        tilepath_generator: Callable[[Path, str, ImageFormat, TileIdentifier], Path],
        rounds: Sequence[int],
        chs: Sequence[int],
        zplanes: Sequence[int],
        tile_height: int,
        tile_width: int,
        x_coords: Tuple[int, int],
        y_coords: Tuple[int, int],
        zplane_to_coords: Mapping[int, Number],
):
    """Write the tiles for a single aligned (physical coordinates) regular (the dimensions have the
    same cardinality) FOV, with a provided tile_path_generator.  Then build an experiment from the
    tiles.  Finally, load the resulting experiment as an ImageStack and verify that the data
    matches."""
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
        ImageFormat.TIFF,
        tile_identifiers,
        tile_fetcher,
        tilepath_generator,
    )

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


def test_zeropadded_filenames(
        tmpdir,
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
    tmpdir_path = Path(tmpdir)
    def tilepath_generator(
            basepath: Path,
            image_type: str,
            tile_format: ImageFormat,
            tile_identifier: TileIdentifier,
    ) -> Path:
        return basepath / (
            f"{image_type}-"
            f"f{tile_identifier.fov_id}-"
            f"r{tile_identifier.round_label:02}-"
            f"c{tile_identifier.ch_label}-"
            f"z{tile_identifier.zplane_label}."
            f"{tile_format.file_ext}"
        )

    tilepath_test(
        tmpdir_path,
        tilepath_generator,
        rounds,
        chs,
        zplanes,
        tile_height,
        tile_width,
        x_coords,
        y_coords,
        zplane_to_coords,
    )


def test_tiles_in_dirs(
        tmpdir,
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
    tmpdir_path = Path(tmpdir)
    def tilepath_generator(
            basepath: Path,
            image_type: str,
            tile_format: ImageFormat,
            tile_identifier: TileIdentifier,
    ) -> Path:
        rand_subdir = "".join([
            random.choice(string.ascii_letters + string.digits)
            for n in range(16)])
        return basepath / rand_subdir / (
            f"{image_type}-"
            f"f{tile_identifier.fov_id}-"
            f"r{tile_identifier.round_label:02}-"
            f"c{tile_identifier.ch_label}-"
            f"z{tile_identifier.zplane_label}."
            f"{tile_format.file_ext}"
        )

    tilepath_test(
        tmpdir_path,
        tilepath_generator,
        rounds,
        chs,
        zplanes,
        tile_height,
        tile_width,
        x_coords,
        y_coords,
        zplane_to_coords,
    )
