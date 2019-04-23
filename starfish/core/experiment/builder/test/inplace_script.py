import hashlib
import os
import sys
from pathlib import Path
from typing import cast, Mapping, Tuple, Union

import numpy as np
from skimage.io import imread, imsave
from slicedimage import ImageFormat

from starfish.core.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.core.experiment.builder.inplace import (
    enable_inplace_mode, inplace_tile_opener, InplaceFetchedTile
)
from starfish.core.experiment.experiment import Experiment, FieldOfView
from starfish.core.types import Axes, Coordinates, Number


SHAPE = {Axes.Y: 500, Axes.X: 1390}


class InplaceTile(InplaceFetchedTile):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    @property
    def shape(self) -> Mapping[Axes, int]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
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

    def tile_data(self) -> np.ndarray:
        return imread(os.fspath(self.file_path))

    @property
    def filepath(self) -> Path:
        return self.file_path


class InplaceFetcher(TileFetcher):
    def __init__(self, input_dir: Path, prefix: str):
        self.input_dir = input_dir
        self.prefix = prefix

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        filename = '{}-Z{}-H{}-C{}.tiff'.format(self.prefix, z, r, ch)
        file_path = self.input_dir / f"fov_{fov:03}" / filename
        return InplaceTile(file_path)


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

    enable_inplace_mode()

    write_experiment_json(
        path=os.fspath(image_dir),
        fov_count=num_fovs,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        primary_tile_fetcher=InplaceFetcher(image_dir, FieldOfView.PRIMARY_IMAGES),
        aux_tile_fetcher={
            aux_img_name: InplaceFetcher(image_dir, aux_img_name)
            for aux_img_name in aux_name_to_dimensions.keys()
        },
        postprocess_func=add_codebook,
        default_shape=SHAPE,
        fov_path_generator=fov_path_generator,
        tile_opener=inplace_tile_opener,
    )


def write_image(
        num_fovs: int,
        image_dimensions: Mapping[Union[Axes, str], int],
        fetcher: InplaceFetcher
):
    for fov_num in range(num_fovs):
        for r in range(image_dimensions[Axes.ROUND]):
            for ch in range(image_dimensions[Axes.CH]):
                for zplane in range(image_dimensions[Axes.ZPLANE]):
                    tile = cast(InplaceFetchedTile, fetcher.get_tile(fov_num, r, ch, zplane))
                    path = tile.filepath
                    path.parent.mkdir(parents=True, exist_ok=True)

                    data = np.random.random(size=(SHAPE[Axes.Y], SHAPE[Axes.X]))

                    imsave(os.fspath(path), data, plugin="tifffile")


def write_inplace(path: str, num_fovs: int = 2):
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

    tmpdir = Path(path)

    # write out the image files
    primary_fetcher = InplaceFetcher(tmpdir, FieldOfView.PRIMARY_IMAGES)
    write_image(num_fovs, primary_image_dimensions, primary_fetcher)

    for aux_img_name in aux_name_to_dimensions.keys():
        aux_fetcher = InplaceFetcher(tmpdir, aux_img_name)
        write_image(num_fovs, aux_name_to_dimensions[aux_img_name], aux_fetcher)

    # format the experiment.
    format_data(tmpdir, primary_image_dimensions, aux_name_to_dimensions, num_fovs)

    Experiment.from_json(os.fspath(tmpdir / "experiment.json"))


if __name__ == "__main__":
    write_inplace(sys.argv[1])
