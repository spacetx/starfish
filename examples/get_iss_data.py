import argparse
import json
import io
import os
import zipfile
from typing import IO, Tuple

import requests
from slicedimage import ImageFormat

from examples.support import FetchedImage, ImageFetcher, write_experiment_json
from starfish.constants import Indices, Features
from starfish.util.argparse import FsExistsType

SHAPE = (980, 1330)


def download(input_dir, url):
    print("Downloading data ...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(input_dir)


class ISSImage(FetchedImage):
    def __init__(self, file_path):
        self.file_path = file_path

    @property
    def shape(self) -> Tuple[int, ...]:
        return SHAPE

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def image_data_handle(self) -> IO:
        return open(self.file_path, "rb")


class HybridizationImageFetcher(ImageFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        return ISSImage(os.path.join(self.input_dir, str(hyb + 1), "c{}.TIF".format(ch + 2)))


class AuxImageFetcher(ImageFetcher):
    def __init__(self, path):
        self.path = path

    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        return ISSImage(self.path)


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

        # TODO: (ttung) remove the following unholy hacks.  this is because we want to point at a tileset rather than
        # a collection.
        experiment_json_doc['hybridization_images'] = "hybridization-fov_000.json"
        experiment_json_doc['auxiliary_images']['nuclei'] = "nuclei-fov_000.json"
        experiment_json_doc['auxiliary_images']['dots'] = "dots-fov_000.json"
        return experiment_json_doc

    # the magic numbers here are just for the ISS example data set.
    write_experiment_json(
        output_dir,
        1,
        {
            Indices.ROUND: 4,
            Indices.CH: 4,
            Indices.Z: 1,
        },
        {
            'nuclei': {
                Indices.ROUND: 1,
                Indices.CH: 1,
                Indices.Z: 1,
            },
            'dots': {
                Indices.ROUND: 1,
                Indices.CH: 1,
                Indices.Z: 1,
            }
        },
        hyb_image_fetcher=HybridizationImageFetcher(input_dir),
        aux_image_fetcher={
            'nuclei': AuxImageFetcher(os.path.join(input_dir, "DO", "c1.TIF")),
            'dots': AuxImageFetcher(os.path.join(input_dir, "DO", "c2.TIF")),
        },
        postprocess_func=add_codebook,
        default_shape=SHAPE,
    )

    codebook = [
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 2, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 3, Indices.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_human"
        },
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 2, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 3, Indices.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_mouse"
        },
    ]
    codebook_json_filename = "codebook.json"
    write_json(codebook, os.path.join(output_dir, codebook_json_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())
    parser.add_argument("--d", help="Download data", type=bool)

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir, args.d)
