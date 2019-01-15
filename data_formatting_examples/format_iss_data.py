import argparse
import io
import json
import os
import zipfile
from typing import Mapping, Tuple, Union

import numpy as np
import requests
from skimage.io import imread
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.experiment.builder import write_experiment_json
from starfish.types import Axes, Coordinates, Features, Number
from starfish.util.argparse import FsExistsType

SHAPE = 980, 1330


class ISSTile(FetchedTile):
    def __init__(self, file_path):
        self.file_path = file_path

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
        return imread(self.file_path)


class ISSPrimaryTileFetcher(TileFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        return ISSTile(os.path.join(self.input_dir, str(r + 1), "c{}.TIF".format(ch + 2)))


class ISSAuxTileFetcher(TileFetcher):
    def __init__(self, path):
        self.path = path

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        return ISSTile(self.path)


def download(input_dir, url):
    print("Downloading data ...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(input_dir)


def write_json(res, output_path):
    json_doc = json.dumps(res, indent=4)
    print(json_doc)
    print("Writing to: {}".format(output_path))
    with open(output_path, "w") as outfile:
        json.dump(res, outfile, indent=4)


def format_data(input_dir, output_dir, d):
    if not input_dir.endswith("/"):
        input_dir += "/"

    if not output_dir.endswith("/"):
        output_dir += "/"

    if d:
        url = "http://d1zymp9ayga15t.cloudfront.net/content/Examplezips/ExampleInSituSequencing.zip"
        download(input_dir, url)
        input_dir += "ExampleInSituSequencing/"
        print("Data downloaded to: {}".format(input_dir))
    else:
        input_dir += "ExampleInSituSequencing/"
        print("Using data in : {}".format(input_dir))

    def add_codebook(experiment_json_doc):
        experiment_json_doc['codebook'] = "codebook.json"

        return experiment_json_doc

    # the magic numbers here are just for the ISS example data set.
    write_experiment_json(
        output_dir,
        1,
        ImageFormat.TIFF,
        primary_image_dimensions={
            Axes.ROUND: 4,
            Axes.CH: 4,
            Axes.ZPLANE: 1,
        },
        aux_name_to_dimensions={
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
        },
        primary_tile_fetcher=ISSPrimaryTileFetcher(input_dir),
        aux_tile_fetcher={
            'nuclei': ISSAuxTileFetcher(os.path.join(input_dir, "DO", "c1.TIF")),
            'dots': ISSAuxTileFetcher(os.path.join(input_dir, "DO", "c2.TIF")),
        },
        postprocess_func=add_codebook,
        default_shape=SHAPE
    )

    codebook_array = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 3, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 3, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_human"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 3, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_mouse"
        },
    ]
    codebook = Codebook.from_code_array(codebook_array)
    codebook_json_filename = "codebook.json"
    codebook.to_json(os.path.join(output_dir, codebook_json_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())
    parser.add_argument("--d", help="Download data", type=bool)

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir, args.d)
