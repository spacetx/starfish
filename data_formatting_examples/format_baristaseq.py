"""
This script contains TileFetchers and FetchedTile objects to convert data donated by the Zador lab
to SpaceTx format. The locations of the data files for use with this script can be found in the
docstring for `format_data`
"""

import os
import shutil
from typing import Mapping, Tuple, Union

import click
import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat

from starfish.experiment.builder import (FetchedTile, TileFetcher,
                                         write_experiment_json)
from starfish.types import Axes, Coordinates, Number

DEFAULT_TILE_SHAPE = 1000, 800


class BaristaSeqTile(FetchedTile):
    def __init__(self, file_path):
        self.file_path = file_path

    @property
    def shape(self) -> Tuple[int, ...]:
        return DEFAULT_TILE_SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        # these are dummy coordinates
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return imread(self.file_path)


class BaristaSeqTileFetcher(TileFetcher):
    def __init__(self, input_dir) -> None:
        self.input_dir = input_dir

    def get_tile(self, fov: int, hyb: int, ch: int, z: int) -> FetchedTile:
        subdir = "primary"
        round_dir = f"r{hyb}"
        if hyb == 0:
            filename = f"T{fov+1:05}C{ch+1:02}Z{z+1:03}.tif"
        else:
            filename = f"alignedT{fov+1:05}C{ch+1:02}Z{z+1:03}.tif"
        file_path = os.path.join(self.input_dir, subdir, round_dir, filename)
        return BaristaSeqTile(file_path)


class BaristaSeqNucleiTileFetcher(TileFetcher):
    def __init__(self, input_dir, aux_type) -> None:
        self.input_dir = input_dir

    def get_tile(self, fov: int, hyb: int, ch: int, z: int) -> FetchedTile:
        subdir = "nissl"
        filename = f"T00001C05Z{z+1:03}.tif"
        file_path = os.path.join(self.input_dir, subdir, filename)

        return BaristaSeqTile(file_path)


@click.command()
@click.option("--input-dir", type=str, required=True, help="input directory containing images")
@click.option("--output-dir", type=str, required=True, help="output directory for formatted data")
def format_data(input_dir, output_dir) -> None:
    """Format a BaristaSeq Tile

    Parameters
    ----------
    input_dir : str
        Input directory containing data. Example data for a single FoV can be downloaded from
        s3://spacetx.starfish.data.public/browse/raw/20181231/barista-seq-mouse-cortex-cropped
    output_dir : str
        Output directory containing formatted data in SpaceTx format. Example output data can be
        downloaded from
        https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20181028/ \
        BaristaSeq/cropped_formatted/experiment.json"
    """

    num_fovs = 1

    primary_image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: 3,
        Axes.CH: 4,
        Axes.ZPLANE: 17,
    }

    aux_name_to_dimensions: Mapping[str, Mapping[Union[str, Axes], int]] = {
        "nuclei": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 17,
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    write_experiment_json(
        path=output_dir,
        fov_count=num_fovs,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        primary_tile_fetcher=BaristaSeqTileFetcher(input_dir),
        aux_tile_fetcher={
            "nuclei": BaristaSeqNucleiTileFetcher(input_dir, "nuclei"),
        },
        tile_format=ImageFormat.TIFF,
        default_shape=DEFAULT_TILE_SHAPE
    )

    shutil.copyfile(
        src=os.path.join(input_dir, "codebook.json"),
        dst=os.path.join(output_dir, "codebook.json")
    )


if __name__ == "__main__":
    format_data()
