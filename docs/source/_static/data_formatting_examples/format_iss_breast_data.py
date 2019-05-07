"""
Format In-Situ Sequencing Data
==============================

The following script formats In-Situ Sequencing data in SpaceTx-Format
"""
import argparse
import os
from typing import Mapping, Tuple, Union

import numpy as np
import pandas as pd
from skimage.io import imread
from slicedimage import ImageFormat

from starfish.core.util.argparse import FsExistsType
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, Number


class IssCroppedBreastTile(FetchedTile):
    def __init__(self, file_path, coordinates):
        self.file_path = file_path
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 1044, Axes.X: 1390}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return self._coordinates

    @staticmethod
    def crop(img):
        crp = img[40:1084, 20:1410]
        return crp

    def tile_data(self) -> np.ndarray:
        return self.crop(imread(self.file_path))


class ISSCroppedBreastPrimaryTileFetcher(TileFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        coordinates = os.path.join(input_dir, "coordinates.csv")
        self.coordinates_df = pd.read_csv(coordinates, index_col=0)

    @property
    def ch_dict(self):
        ch_dict = {0: 'FITC', 1: 'Cy3', 2: 'Cy3 5', 3: 'Cy5'}
        return ch_dict

    @property
    def round_dict(self):
        round_str = ['1st', '2nd', '3rd', '4th']
        round_dict = dict(enumerate(round_str))
        return round_dict

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:

        # get filepath
        fov_ = str(fov + 1)
        round_ = self.round_dict[r]
        ch_ = self.ch_dict[ch]
        filename = f"slideA_{fov_}_{round_}_{ch_}.TIF"
        file_path = os.path.join(self.input_dir, filename)

        # get coordinates
        coordinates = {
            Coordinates.X: (
                self.coordinates_df.loc[fov, "x_min"],
                self.coordinates_df.loc[fov, "x_max"]
            ),
            Coordinates.Y: (
                self.coordinates_df.loc[fov, "y_min"],
                self.coordinates_df.loc[fov, "y_max"]
            ),
            Coordinates.Z: 0
        }

        return IssCroppedBreastTile(file_path, coordinates)


class ISSCroppedBreastAuxTileFetcher(TileFetcher):
    def __init__(self, input_dir, aux_type):
        self.input_dir = input_dir
        self.aux_type = aux_type
        coordinates = os.path.join(input_dir, "coordinates.csv")
        self.coordinates_df = pd.read_csv(coordinates, index_col=0)

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        if self.aux_type == 'nuclei':
            filename = 'slideA_{}_DO_DAPI.TIF'.format(str(fov + 1))
        elif self.aux_type == 'dots':
            filename = 'slideA_{}_DO_Cy3.TIF'.format(str(fov + 1))
        else:
            msg = 'invalid aux type: {}'.format(self.aux_type)
            msg += ' expected either nuclei or dots'
            raise ValueError(msg)

        file_path = os.path.join(self.input_dir, filename)

        # get coordinates
        coordinates = {
            Coordinates.X: (
                self.coordinates_df.loc[fov, "x_min"],
                self.coordinates_df.loc[fov, "x_max"]
            ),
            Coordinates.Y: (
                self.coordinates_df.loc[fov, "y_min"],
                self.coordinates_df.loc[fov, "y_max"]
            ),
            Coordinates.Z: 0
        }

        return IssCroppedBreastTile(file_path, coordinates=coordinates)


def format_data(input_dir, output_dir):

    primary_image_dimensions = {
        Axes.ROUND: 4,
        Axes.CH: 4,
        Axes.ZPLANE: 1,
    }

    aux_name_to_dimensions = {
        'nuclei': {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
        'dots': {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        }
    }

    write_experiment_json(
        path=output_dir,
        fov_count=16,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        primary_tile_fetcher=ISSCroppedBreastPrimaryTileFetcher(input_dir),
        aux_tile_fetcher={
            'nuclei': ISSCroppedBreastAuxTileFetcher(input_dir, 'nuclei'),
            'dots': ISSCroppedBreastAuxTileFetcher(input_dir, 'dots'),
        },
    )


if __name__ == "__main__":
    """
    This TileFetcher should be run on data found at:
    s3://spacetx.starfish.data/mignardi_breast_1/raw/

    The data produced by this TileFetcher have been uploaded and can be found at the following
    prefix:
    s3://spacetx.starfish.data.public/browse/formatted/iss/20190506
    """

    s3_bucket = "s3://czi.starfish.data.public/browse/raw/20180820/iss_breast/"
    input_help_msg = "Path to raw data. Raw data can be downloaded from: {}".format(s3_bucket)
    output_help_msg = "Path to output experment.json and all formatted images it references"
    fov_help_msg = "The number of fovs that should be extracted from the directory"
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType(), help=input_help_msg)
    parser.add_argument("output_dir", type=FsExistsType(), help=output_help_msg)

    args = parser.parse_args()
    format_data(args.input_dir, args.output_dir)
