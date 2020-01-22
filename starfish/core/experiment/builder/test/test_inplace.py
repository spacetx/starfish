import hashlib
import os
from pathlib import Path
from typing import Mapping, Union

import numpy as np
from skimage.io import imsave
from slicedimage import ImageFormat

from starfish.core.types import Axes, Coordinates, CoordinateValue
from ..builder import write_experiment_json
from ..inplace import (
    InplaceFetchedTile, InplaceWriterContract,
)
from ..providers import FetchedTile, TileFetcher
from ...experiment import Experiment, FieldOfView


SHAPE = {Axes.Y: 500, Axes.X: 1390}


def test_inplace(tmpdir):
    tmpdir_path = Path(tmpdir)

    write_inplace(tmpdir_path)

    # load up the experiment, and select an image.  Ensure that it has non-zero data.  This is to
    # verify that we are sourcing the data from the tiles that were already on-disk, and not the
    # artificially zero'ed tiles that we feed the experiment builder.
    experiment = Experiment.from_json(os.fspath(tmpdir_path / "experiment.json"))
    primary_image = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
    assert not np.allclose(primary_image.xarray, 0)


def tile_fn(input_dir: Path, prefix: str, fov: int, r: int, ch: int, zplane: int) -> Path:
    filename = '{}-Z{}-H{}-C{}.tiff'.format(prefix, zplane, r, ch)
    return input_dir / f"fov_{fov:03}" / filename


class ZeroesInplaceTile(InplaceFetchedTile):
    """These tiles contain all zeroes.  This is irrelevant to the actual experiment construction
    because we are using in-place mode.  That means we build references to the files already on-disk
    and any data returned is merely metadata (tile shape, tile coordinates, and tile checksum."""
    def __init__(self, file_path: Path):
        self.file_path = file_path

    @property
    def shape(self) -> Mapping[Axes, int]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def sha256(self) -> str:
        hasher = hashlib.sha256()
        with open(str(self.file_path), "rb") as fh:
            hasher.update(fh.read())
        return hasher.hexdigest()

    @property
    def filepath(self) -> Path:
        return self.file_path


class ZeroesInplaceFetcher(TileFetcher):
    def __init__(self, input_dir: Path, prefix: str):
        self.input_dir = input_dir
        self.prefix = prefix

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = '{}-Z{}-H{}-C{}.tiff'.format(self.prefix, zplane_label, round_label, ch_label)
        file_path = self.input_dir / f"fov_{fov_id:03}" / filename
        return ZeroesInplaceTile(file_path)


def fov_path_generator(parent_toc_path: Path, toc_name: str) -> Path:
    return parent_toc_path.parent / toc_name / "{}.json".format(parent_toc_path.stem)


def format_data(
        image_dir: Path,
        primary_image_dimensions: Mapping[Union[Axes, str], int],
        aux_name_to_dimensions: Mapping[str, Mapping[Union[Axes, str], int]],
        num_fovs):
    def add_codebook(experiment_json_doc):
        experiment_json_doc['codebook'] = "codebook.json"
        return experiment_json_doc

    write_experiment_json(
        path=os.fspath(image_dir),
        fov_count=num_fovs,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        primary_tile_fetcher=ZeroesInplaceFetcher(image_dir, FieldOfView.PRIMARY_IMAGES),
        aux_tile_fetcher={
            aux_img_name: ZeroesInplaceFetcher(image_dir, aux_img_name)
            for aux_img_name in aux_name_to_dimensions.keys()
        },
        postprocess_func=add_codebook,
        default_shape=SHAPE,
        writer_contract=InplaceWriterContract(),
    )


def write_image(
        base_path: Path,
        prefix: str,
        num_fovs: int,
        image_dimensions: Mapping[Union[Axes, str], int],
):
    """Writes the constituent tiles of an image to disk.  The tiles are made up with random noise.
    """
    for fov_num in range(num_fovs):
        for round_label in range(image_dimensions[Axes.ROUND]):
            for ch_label in range(image_dimensions[Axes.CH]):
                for zplane_label in range(image_dimensions[Axes.ZPLANE]):
                    path = tile_fn(base_path, prefix, fov_num, round_label, ch_label, zplane_label)
                    path.parent.mkdir(parents=True, exist_ok=True)

                    data = np.random.random(size=(SHAPE[Axes.Y], SHAPE[Axes.X])).astype(np.float32)

                    imsave(os.fspath(path), data, plugin="tifffile")


def write_inplace(tmpdir: Path, num_fovs: int = 2):
    primary_image_dimensions: Mapping[Union[Axes, str], int] = {
        Axes.ROUND: 4,
        Axes.CH: 4,
        Axes.ZPLANE: 1,
    }

    aux_name_to_dimensions: Mapping[str, Mapping[Union[Axes, str], int]] = {
        'nuclei': {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
    }

    # write out the image files
    write_image(tmpdir, FieldOfView.PRIMARY_IMAGES, num_fovs, primary_image_dimensions)

    for aux_img_name in aux_name_to_dimensions.keys():
        write_image(tmpdir, aux_img_name, num_fovs, aux_name_to_dimensions[aux_img_name])

    # format the experiment.
    format_data(tmpdir, primary_image_dimensions, aux_name_to_dimensions, num_fovs)

    Experiment.from_json(os.fspath(tmpdir / "experiment.json"))
