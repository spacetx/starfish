import csv
import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, MutableSequence, Optional, Sequence

import numpy as np
from slicedimage import ImageFormat

from starfish.core.types import Axes, Coordinates, CoordinateValue
from ..factories import unique_data
from ... import FetchedTile, TileFetcher, TileIdentifier
from ...structured_formatter import TILE_COORDINATE_NAMES


def format_data(
        image_directory_path: Path,
        coordinates_csv_path: Path,
        output_path: Path,
        tile_format: ImageFormat,
        in_place: bool,
) -> None:
    """Inplace experiment construction monkeypatches the code destructively.  To isolate these side
    effects, we run the experiment construction in a separate process."""
    this = Path(__file__)
    structured_formatter_script_path = this.parent / "structured_formatter_script.py"

    subprocess.check_call(
        [sys.executable,
         os.fspath(structured_formatter_script_path),
         os.fspath(image_directory_path),
         os.fspath(coordinates_csv_path),
         os.fspath(output_path),
         tile_format.name,
         str(in_place),
         ]
    )


class UniqueTiles(FetchedTile):
    """Tiles where the pixel values are unique per fov/round/ch/z."""
    def __init__(
            self,
            # these are the arguments passed in as a result of tile_fetcher_factory's
            # pass_tile_indices parameter.
            fov_id: int, round_id: int, ch_id: int, zplane_id: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            fovs: Sequence[int], rounds: Sequence[int], chs: Sequence[int], zplanes: Sequence[int],
            tile_height: int, tile_width: int,
    ) -> None:
        super().__init__()
        self.fov_id = fov_id
        self.round_id = round_id
        self.ch_id = ch_id
        self.zplane_id = zplane_id
        self.fovs = fovs
        self.rounds = rounds
        self.chs = chs
        self.zplanes = zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self.tile_height, Axes.X: self.tile_width}

    def tile_data(self) -> np.ndarray:
        """Return the data for a given tile."""
        return unique_data(
            self.fov_id,
            self.rounds.index(self.round_id),
            self.chs.index(self.ch_id),
            self.zplanes.index(self.zplane_id),
            len(self.fovs),
            len(self.rounds),
            len(self.chs),
            len(self.zplanes),
            self.tile_height,
            self.tile_width)


def write_tile_data(
        basepath: Path,
        image_type: str,
        tile_format: ImageFormat,
        tile_identifiers: Sequence[TileIdentifier],
        fetcher: TileFetcher,
        tilepath_generator: Optional[Callable[
            [Path, str, ImageFormat, TileIdentifier], Path]] = None,
) -> None:
    if tilepath_generator is None:
        tilepath_generator = \
            lambda _basepath, _image_type, _tile_format, _tile_identifier: basepath / (
                f"{_image_type}-"
                f"f{_tile_identifier.fov_id}-"
                f"r{_tile_identifier.round_label}-"
                f"c{_tile_identifier.ch_label}-"
                f"z{_tile_identifier.zplane_label}."
                f"{tile_format.file_ext}"
            )

    for tile_identifier in tile_identifiers:
        fetched_tile = fetcher.get_tile(
            tile_identifier.fov_id,
            tile_identifier.round_label,
            tile_identifier.ch_label,
            tile_identifier.zplane_label)

        tilepath = tilepath_generator(basepath, image_type, tile_format, tile_identifier)
        tilepath.parent.mkdir(parents=True, exist_ok=True)

        tile_format.writer_func(tilepath, fetched_tile.tile_data())


def render_coordinates_to_rows(
        tile_to_physical_coordinates: Mapping[
            TileIdentifier, Mapping[Coordinates, CoordinateValue]],
) -> Sequence[Mapping[str, str]]:
    results: MutableSequence[Mapping[str, str]] = list()

    for tile_identifier, physical_coordinates in tile_to_physical_coordinates.items():
        rowdata: MutableMapping[str, str] = dict()

        for tile_coordinate_name, tile_coordinate_value in zip(
                TILE_COORDINATE_NAMES, dataclasses.astuple(tile_identifier)
        ):
            rowdata[tile_coordinate_name] = str(tile_coordinate_value)

        for coordinate_name in list(Coordinates):
            coordinate_value = physical_coordinates.get(coordinate_name, None)
            if coordinate_value is None and coordinate_name == Coordinates.Z:
                # Z coordinates may be legitimately missing
                continue

            if isinstance(coordinate_value, tuple):
                rowdata[f'{coordinate_name}_min'] = str(coordinate_value[0])
                rowdata[f'{coordinate_name}_max'] = str(coordinate_value[1])
            else:
                rowdata[f'{coordinate_name}_min'] = str(coordinate_value)

        results.append(rowdata)

    return results


def write_coordinates_csv(
        path: Path,
        rows: Sequence[Mapping[str, str]],
        write_z_coordinates_in_header: bool,
) -> None:
    headers = list(TILE_COORDINATE_NAMES)
    for coordinate_name in list(Coordinates):
        if coordinate_name == Coordinates.Z and not write_z_coordinates_in_header:
            continue
        headers.append(f"{coordinate_name.value}_min")
        headers.append(f"{coordinate_name.value}_max")

    with open(os.fspath(path), "w") as fh:
        writer = csv.DictWriter(fh, headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
