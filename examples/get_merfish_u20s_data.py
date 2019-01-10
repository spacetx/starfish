import argparse
import functools
import os
from typing import IO, Mapping, Tuple, Union

import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat

from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, Number
from starfish.util.argparse import FsExistsType

SHAPE = 2048, 2048


# We use this to cache images across tiles.  In the case of the merfish data set, FOVs are saved
# together in a single file.  To avoid reopening and decoding the TIFF file, we use a single-element
# cache that maps between file_path and the decoded multipage TIFF.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path):
    return imread(file_path)


class MERFISHTile(FetchedTile):
    def __init__(self, file_path, r, ch):
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

    @property
    def shape(self) -> Tuple[int, ...]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        # FIXME: (dganguli) please provide proper coordinates here.
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    def tile_data(self) -> IO:
        return cached_read_fn(self.file_path)[self.map[(self.r, self.ch)], :, :]


class MERFISHAuxTile(FetchedTile):
    def __init__(self, file_path):
        self.file_path = file_path
        self.dapi_index = 17

    @property
    def shape(self) -> Tuple[int, ...]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        # FIXME: (dganguli) please provide proper coordinates here.
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.dapi_index, :, :]


class MERFISHTileFetcher(TileFetcher):
    def __init__(self, input_dir, is_dapi):
        self.input_dir = input_dir
        self.is_dapi = is_dapi

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        filename = os.path.join(self.input_dir, 'fov_{}.tif'.format(fov))
        file_path = os.path.join(self.input_dir, filename)
        if self.is_dapi:
            return MERFISHAuxTile(file_path)
        else:
            return MERFISHTile(file_path, r, ch)


def format_data(input_dir, output_dir):
    def add_codebook(experiment_json_doc):
        experiment_json_doc['codebook'] = "codebook.json"
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
                          postprocess_func=add_codebook,
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
