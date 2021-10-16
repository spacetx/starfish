"""
.. _format_merfish:

Format MERFISH data
===================

The following script formats MERFISH data acquired from cultured U2-OS cells.
This is a good example of:

* converting multipage TIFF files that contain every round, channel of a FOV in a single file
* multiple fields of view (FOV)
* fetching primary images and auxiliary images from the same file
* assigning physical coordinates from CSV file

.. note::

   This example is provided for illustrative purposes, demonstrating how the
   :py:class:`.TileFetcher` is used in practice. It will need to be adapted to meet
   the specific needs of your data.

The experiment had 496 of fields of view. Each FOV is stored in one file consisting of primary
images (8 rounds, 2 channels, 1 z-tile) and one DAPI-stained image. The tile index that maps to
each round and channel is defined in the :py:class:`.FetchedTile` subclasses.

input data structure:
::

    └── parent
        ├── stagePos.csv
        ├── fov_0.tif
        ├── fov_1.tif
        ├── fov_2.tif
        ├── ...

The locations of the data files for use with this script can be found in the s3_bucket variable.
"""
import argparse
import functools
import json
import os
from typing import Mapping, Union

import numpy as np
import pandas as pd
from skimage.io import imread
from slicedimage import ImageFormat

from starfish.core.util.argparse import FsExistsType
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue

SHAPE = {Axes.Y: 2048, Axes.X: 2048}


# We use this to cache images across tiles.  In the case of the merfish data set, FOVs are saved
# together in a single file.  To avoid reopening and decoding the TIFF file, we use a single-element
# cache that maps between file_path and the decoded multipage TIFF.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path):
    return imread(file_path)


class MERFISHTile(FetchedTile):
    def __init__(self, file_path, r, ch, coordinates):
        self.file_path = file_path
        # how to index tiles from indices into multi-page tiff
        # key is a tuple of round, chan. val is the index
        self.map = {(0, 0): 0,
                    (0, 1): 1,
                    (1, 0): 3,
                    (1, 1): 2,
                    (2, 0): 4,
                    (2, 1): 5,
                    (3, 0): 6,
                    (3, 1): 7,
                    (4, 0): 9,
                    (4, 1): 8,
                    (5, 0): 10,
                    (5, 1): 11,
                    (6, 0): 12,
                    (6, 1): 13,
                    (7, 0): 15,
                    (7, 1): 14}
        self.r = r
        self.ch = ch
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return self._coordinates

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.map[(self.r, self.ch)], :, :]


class MERFISHAuxTile(FetchedTile):
    def __init__(self, file_path, coordinates):
        self.file_path = file_path
        self.dapi_index = 17
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return self._coordinates

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.dapi_index, :, :]


class MERFISHTileFetcher(TileFetcher):
    def __init__(self, input_dir, is_dapi):
        self.input_dir = input_dir
        self.is_dapi = is_dapi
        self.coordinates = self.parse_coordinates(input_dir)

    @staticmethod
    def parse_coordinates(input_dir):
        filename = os.path.join(input_dir, "stagePos.csv")
        data = pd.read_csv(filename, names=['y_min', 'x_min'])
        data['x_max'] = data['x_min'] + 200
        data['y_max'] = data['y_min'] + 200
        return data

    def make_coordinates(self, fov):
        return {
            Coordinates.X: (
                float(self.coordinates.loc[fov, 'x_min']),
                float(self.coordinates.loc[fov, 'x_max'])
            ),
            Coordinates.Y: (
                float(self.coordinates.loc[fov, 'y_min']),
                float(self.coordinates.loc[fov, 'y_max'])
            ),
            Coordinates.Z: (0.0, 0.001)
        }

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = os.path.join(self.input_dir, 'fov_{}.tif'.format(fov_id))
        file_path = os.path.join(self.input_dir, filename)
        if self.is_dapi:
            return MERFISHAuxTile(file_path, self.make_coordinates(fov_id))
        else:
            return MERFISHTile(file_path, round_label, ch_label, self.make_coordinates(fov_id))


def format_data(input_dir, output_dir):

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    def add_scale_factors(experiment_json_doc):
        filename = os.path.join(input_dir, "scale_factors.json")
        with open(filename, 'r') as f:
            data = json.load(f)
        experiment_json_doc['extras'] = {"scale_factors": data}
        return experiment_json_doc

    num_fovs = 496

    primary_image_dimensions = {
        Axes.ROUND: 8,
        Axes.CH: 2,
        Axes.ZPLANE: 1,
    }

    aux_name_to_dimensions = {
        'nuclei': {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1
        }
    }

    write_experiment_json(output_dir,
                          num_fovs,
                          tile_format=ImageFormat.TIFF,
                          primary_image_dimensions=primary_image_dimensions,
                          aux_name_to_dimensions=aux_name_to_dimensions,
                          primary_tile_fetcher=MERFISHTileFetcher(input_dir, is_dapi=False),
                          aux_tile_fetcher={
                              'nuclei': MERFISHTileFetcher(input_dir, is_dapi=True),
                          },
                          postprocess_func=add_scale_factors,
                          default_shape=SHAPE
                          )


if __name__ == "__main__":
    s3_bucket = "s3://czi.starfish.data.public/browse/raw/20180820/merfish_u2os/"
    input_help_msg = "Path to raw data. Raw data can be downloaded from: {}".format(s3_bucket)
    output_help_msg = "Path to output experment.json and all formatted images it references"
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType(), help=input_help_msg)
    parser.add_argument("output_dir", type=FsExistsType(), help=output_help_msg)

    args = parser.parse_args()
    format_data(args.input_dir, args.output_dir)
