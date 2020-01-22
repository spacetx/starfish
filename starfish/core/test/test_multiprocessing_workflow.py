import itertools
import multiprocessing
import os
from pathlib import Path
from typing import cast, Mapping, Optional, Sequence

from slicedimage import ImageFormat

from starfish.core.experiment.builder.builder import TileIdentifier
from starfish.core.experiment.builder.defaultproviders import tile_fetcher_factory
from starfish.core.experiment.builder.providers import TileFetcher
from starfish.core.experiment.builder.structured_formatter import format_structured_dataset
from starfish.core.experiment.builder.test.factories.unique_tiles import UniqueTiles
from starfish.core.experiment.builder.test.structured_formatter.util import (
    render_coordinates_to_rows,
    write_coordinates_csv,
    write_tile_data,
)
from starfish.core.experiment.experiment import Experiment, FieldOfView
from starfish.core.image import Filter
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Coordinates, CoordinateValue


def test_multiprocessing_workflow(tmpdir):
    exp = build_experiment_with_multiple_unaligned_fovs(Path(tmpdir))
    assert len(exp.fovs()) > 1
    for fov in exp.fovs():
        assert len(list(fov.get_images(FieldOfView.PRIMARY_IMAGES))) > 1

    all_imagestacks = itertools.chain.from_iterable(
        fov.get_images(FieldOfView.PRIMARY_IMAGES)
        for fov in exp.fovs())

    with multiprocessing.Pool(processes=2) as pool:
        output_imagestacks = pool.map(filter_fn, all_imagestacks)

    for output_imagestack in output_imagestacks:
        assert isinstance(output_imagestack, ImageStack)


def filter_fn(imagestack: ImageStack) -> Optional[ImageStack]:
    f = Filter.Clip()
    return f.run(imagestack)


def build_experiment_with_multiple_unaligned_fovs(
        tmpdir_path: Path,
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
        # default values are mutable, but it's for readability reasons!
) -> Experiment:
    """Write the tiles for a multi-fov unaligned (physical coordinates) regular (the dimensions have
    the same cardinality) image.  Then build an experiment from the tiles.  Finally, load the
    resulting experiment and return it."""
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
            Coordinates.X: cast(CoordinateValue, tuple(
                coordinate + float(tile_identifier.round_label) * 0.0001
                for coordinate in fov_to_x_coords[tile_identifier.fov_id])),
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
    return Experiment.from_json(os.fspath(tmpdir_path / "experiment.json"))
