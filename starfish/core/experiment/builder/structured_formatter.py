import collections
import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    cast,
    FrozenSet,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union)
from warnings import warn

import numpy as np
from slicedimage import ImageFormat, WriterContract

from starfish.core.types import Axes, Coordinates, CoordinateValue, Number
from .builder import TileIdentifier, write_irregular_experiment_json
from .inplace import InplaceFetchedTile, InplaceWriterContract
from .providers import FetchedTile, TileFetcher

FILENAME_CRE = re.compile(
    r"(?P<prefix>.+)-f(?P<fov>\d+)-r(?P<round>\d+)-c(?P<ch>\d+)-z(?P<zplane>\d+)\.(?P<ext>.+)")
TILE_COORDINATE_NAMES = ('fov', 'round', 'ch', 'zplane')


class ExtraPhysicalCoordinatesWarning(UserWarning):
    """Raised if there are physical coordinates in the coordinates csv file for a tile identifier
    that is not represented in the tile data.
    """
    pass


class PhysicalCoordinateNotPresentError(Exception):
    """Raised if physical coordinates along an axis is not present.  This is not a user-visible
    error because it may not actually be an error (i.e., exceptions of this class should be caught
    and handled in this module).
    """
    pass


@dataclass
class InferredTileResult:
    identifier: TileIdentifier
    path: Path
    format: ImageFormat


def format_structured_dataset(
        image_directory_path_str: str,
        physical_coordinates_csv_path_str: str,
        output_experiment_dir_str: str,
        tile_format: ImageFormat,
        in_place: bool = False,
) -> None:
    """
    Format a dataset where the filenames of the tiles provide most of the metadata required to
    organize them into the 7D tensor (image_type, fov, round, ch, zplane, y, x).  The remaining
    metadata, namely physical coordinates of the tiles, are provided in a CSV file.

    See the documentation in :ref:`Format Structured Data <format_structured_data>`

    Parameters
    ----------
    image_directory_path_str : str
        The path of the directory where the image tiles are found.
    physical_coordinates_csv_path_str : str
        The path of the csv file containing the physical coordinates.
    output_experiment_dir_str : str
        The path of the directory to write the experiment to.  The top-level experiment json will be
        located at <output_experiment_dir_str>/experiment.json
    tile_format : ImageFormat
        The format of the tiles that are written.  Note that this must match the format of the
        original files if in_place is True.
    in_place : bool
        When this is true, the original tiles are used in the resulting experiment.json.
        (default: False)
    """
    stack_structure: Mapping[str, Sequence[InferredTileResult]] = \
        infer_stack_structure(Path(image_directory_path_str))
    physical_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = \
        read_physical_coordinates_from_csv(Path(physical_coordinates_csv_path_str))

    unseen_tile_coordinates = set(physical_coordinates.keys())

    # we should have physical coordinates for every region in the stack structure.
    for image_type, inferred_tile_results in stack_structure.items():
        for inferred_tile_result in inferred_tile_results:
            if inferred_tile_result.identifier not in physical_coordinates:
                raise ValueError(
                    f"Could not resolve physical coordinates for tile "
                    f"{inferred_tile_result.identifier} in image {image_type}"
                )
            if in_place and inferred_tile_result.format != tile_format:
                raise ValueError(
                    f"Tile {inferred_tile_result.identifier} has a different tile format ("
                    f"{inferred_tile_result.format}) than `format_structured_dataset` was called "
                    f"with ({tile_format}."
                )

            unseen_tile_coordinates.remove(inferred_tile_result.identifier)

    # do we have extra physical coordinates
    for tile_coordinate in unseen_tile_coordinates:
        warn(
            f"physical coordinates for tile {tile_coordinate} provided, but no tile exists at that"
            f"location.",
            ExtraPhysicalCoordinatesWarning,
        )

    image_tile_identifiers: Mapping[str, Sequence[TileIdentifier]] = {
        image_type:
            [inferred_tile_result.identifier for inferred_tile_result in inferred_tile_results]
        for image_type, inferred_tile_results in stack_structure.items()
    }
    tile_fetchers: Mapping[str, TileFetcher] = {
        image_type: InferredTileFetcher(in_place, inferred_tile_results, physical_coordinates)
        for image_type, inferred_tile_results in stack_structure.items()
    }

    if in_place:
        writer_contract: Optional[WriterContract] = InplaceWriterContract()
    else:
        writer_contract = None

    write_irregular_experiment_json(
        output_experiment_dir_str,
        tile_format,
        image_tile_identifiers=image_tile_identifiers,
        tile_fetchers=tile_fetchers,
        writer_contract=writer_contract,
    )


def infer_stack_structure(
        basepath: Path,
) -> Mapping[str, Sequence[InferredTileResult]]:
    results: MutableMapping[str, MutableSequence[InferredTileResult]] = \
        collections.defaultdict(list)

    for path in basepath.glob("**/*"):
        mo = FILENAME_CRE.match(path.name)
        if mo is None:
            continue
        try:
            tile_format = ImageFormat.find_by_extension(mo.group('ext'))
        except ValueError:
            continue

        results[mo.group('prefix')].append(
            InferredTileResult(
                TileIdentifier(*[int(mo.group(component)) for component in TILE_COORDINATE_NAMES]),
                path,
                tile_format,
            )
        )

    return results


def read_physical_coordinates_from_csv(
        csvpath: Path,
) -> Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]]:
    required_coordinates: Mapping[Coordinates, str] = {
        Coordinates.X: "x",
        Coordinates.Y: "y",
    }
    coordinates_required_to_be_tuples: FrozenSet[Coordinates] = frozenset(
        (Coordinates.X, Coordinates.Y))

    result: MutableMapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = dict()
    with open(os.fspath(csvpath), "r") as raw_fh:
        for row_num, row in enumerate(csv.DictReader(raw_fh)):
            try:
                tile_coordinate = TileIdentifier(
                    *[int(row[component]) for component in TILE_COORDINATE_NAMES])
            except KeyError as ex:
                raise ValueError(
                    f"Could not find all the required columns on row {row_num + 1}") from ex

            physical_coordinates: MutableMapping[Coordinates, CoordinateValue] = {}
            for coordinate_name in list(Coordinates):
                try:
                    value = _parse_coordinates(row, coordinate_name)
                except PhysicalCoordinateNotPresentError:
                    # is this a required coordinate?
                    if coordinate_name in required_coordinates:
                        raise ValueError(
                            f"{required_coordinates[coordinate_name]} coordinates not found in row "
                            f"{row_num + 1}")
                    continue

                if (coordinate_name in coordinates_required_to_be_tuples
                        and not isinstance(value, tuple)):
                    raise ValueError(f"{coordinate_name.value} must be a range in row "
                                     f"{row_num + 1}")

                physical_coordinates[coordinate_name] = value

            result[tile_coordinate] = physical_coordinates

    return result


class InferredTile(InplaceFetchedTile):
    INPLACE_SHAPE: Optional[Mapping[Axes, int]] = None

    def __init__(
            self,
            in_place: bool,
            inferred_tile_result: InferredTileResult,
            tile_physical_coordinates:
            Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]],
    ) -> None:
        self.in_place = in_place
        self.inferred_tile_result = inferred_tile_result
        self.tile_physical_coordinates = tile_physical_coordinates
        self._tile_data: Optional[np.ndarray] = None

    @property
    def filepath(self) -> Path:
        return self.inferred_tile_result.path

    def _ensure_tile_loaded(self):
        if self._tile_data is None:
            if self.in_place and InferredTile.INPLACE_SHAPE is not None:
                self._tile_data = np.zeros(
                    (InferredTile.INPLACE_SHAPE[Axes.Y], InferredTile.INPLACE_SHAPE[Axes.X]),
                    dtype=np.float32,
                )
            else:
                self._tile_data = self.inferred_tile_result.format.reader_func(
                    os.fspath(self.inferred_tile_result.path))
                if self.in_place:
                    InferredTile.INPLACE_SHAPE = {
                        Axes.Y: self._tile_data.shape[0],
                        Axes.X: self._tile_data.shape[1],
                    }

    @property
    def sha256(self):
        with open(os.fspath(self.inferred_tile_result.path), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    @property
    def shape(self) -> Mapping[Axes, int]:
        tile_data = self.tile_data()
        return {Axes.Y: tile_data.shape[0], Axes.X: tile_data.shape[1]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return self.tile_physical_coordinates

    def tile_data(self) -> np.ndarray:
        self._ensure_tile_loaded()
        assert self._tile_data is not None
        return self._tile_data


class InferredTileFetcher(TileFetcher):
    def __init__(
            self,
            in_place: bool,
            inferred_tile_results: Sequence[InferredTileResult],
            physical_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]],
    ):
        self.in_place = in_place
        self.inferred_tile_results = inferred_tile_results
        self.physical_coordinates = physical_coordinates

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        wanted_tile_identifier = TileIdentifier(fov, r, ch, z)
        physical_coordinates = cast(
            Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]],
            self.physical_coordinates[wanted_tile_identifier])

        for inferred_tile_result in self.inferred_tile_results:
            if wanted_tile_identifier == inferred_tile_result.identifier:
                return InferredTile(self.in_place, inferred_tile_result, physical_coordinates)

        raise KeyError(f"Could not find data for {wanted_tile_identifier}")


def _convert_str_to_Number(value_str: str) -> Number:
    """Converts a string into a Number.  Conversions to integers are preferred, and if that fails,
    we attempt to convert to a float."""
    try:
        return int(value_str)
    except ValueError:
        pass
    return float(value_str)


def _parse_coordinates(row, coordinate: Coordinates) -> CoordinateValue:
    """Given a row that may contain either no entries, a single scalar value, or a range for
    physical coordinates along an axis, attempt to return a CoordinateValue representing the input.
    In the case where no entries exist, :py:class:`PhysicalCoordinateNotPresentError` is raised.

    The columns will be denoted as the Coordinate value name, suffixed with `_min` or `_max`.  If
    both are present, then we interpret that as a range.  If only one is present, then we interpret
    that as a scalar value.
    """
    prefix = coordinate.value

    min_value = row.get(f"{prefix}_min", None)
    max_value = row.get(f"{prefix}_max", None)

    # convert empty or empty-ish strings into None.
    if min_value is not None and len(min_value.strip()) == 0:
        min_value = None
    if max_value is not None and len(max_value.strip()) == 0:
        max_value = None

    if min_value is not None and max_value is not None:
        return _convert_str_to_Number(min_value), _convert_str_to_Number(max_value)
    elif min_value is not None:
        return _convert_str_to_Number(min_value)
    elif max_value is not None:
        return _convert_str_to_Number(max_value)
    else:
        raise PhysicalCoordinateNotPresentError("neither present")
