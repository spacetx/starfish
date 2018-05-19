import argparse
import io
import os
import json
import zipfile

import requests
from skimage.io import imread, imsave
from slicedimage import ImageFormat, Tile, TileSet, Writer

from starfish.constants import Coordinates, Indices
from starfish.util.argparse import FsExistsType


def download(input_dir, url):
    print("Downloading data ...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(input_dir)


def build_hybridization_stack(input_dir):
    default_shape = imread(os.path.join(input_dir, str(1), "c{}.TIF".format(1))).shape

    hybridization_images = TileSet(
        [Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH],
        {Indices.HYB: 4, Indices.CH: 4},
        default_shape,
        ImageFormat.TIFF,
    )

    for hyb in range(4):
        for ch in range(4):
            tile = Tile(
                {
                    Coordinates.X: (0.0, 0.0001),
                    Coordinates.Y: (0.0, 0.0001),
                },
                {
                    Indices.HYB: hyb,
                    Indices.CH: ch,
                },
            )
            path = os.path.join(input_dir, str(hyb + 1), "c{}.TIF".format(ch + 2))
            tile.set_source_fh_contextmanager(
                lambda _path=path: open(_path, "rb"),
                ImageFormat.TIFF,
            )
            hybridization_images.add_tile(tile)

    return hybridization_images


def build_fov(input_dir, hybridization_stack_name, codebook_json_filename, output_dir):
    prefix = "fov_0"

    nuclei = imread(input_dir + "DO/c1.TIF")
    dots = imread(input_dir + "DO/c2.TIF")

    experiment = {
        'version': "0.0.0",
        'hybridization_images': hybridization_stack_name,
        'codebook': codebook_json_filename,
        'auxiliary_images': {},
    }

    nuclei_fname = "{}_{}.tiff".format(prefix, "nuclei")
    imsave(output_dir + nuclei_fname, nuclei)
    experiment['auxiliary_images']['nuclei'] = {
        'file': nuclei_fname,
        'tile_shape': nuclei.shape,
        'tile_format': "TIFF",
        'coordinates': {
            'x': (0.0, 0.0001),
            'y': (0.0, 0.0001),
        },
    }

    dots_fname = "{}_{}.tiff".format(prefix, "dots")
    imsave(output_dir + dots_fname, dots)
    experiment['auxiliary_images']['dots'] = {
        'file': dots_fname,
        'tile_shape': dots.shape,
        'tile_format': "TIFF",
        'coordinates': {
            'x': (0.0, 0.0001),
            'y': (0.0, 0.0001),
        },
    }

    return experiment


def write_json(res, output_path):
    json_doc = json.dumps(res, indent=4)
    print(json_doc)
    print("Writing to: {}".format(output_path))
    with open(output_path, "w") as outfile:
        json.dump(res, outfile, indent=4)


def tile_opener(tileset_path, tile, ext):
    tile_basename = os.path.splitext(tileset_path)[0]
    return open(
        "{}-H{}-C{}.{}".format(
            tile_basename,
            tile.indices[Indices.HYB],
            tile.indices[Indices.CH],
            "tiff",  # this is not `ext` because ordinarily, output is saved as .npy.  since we're copying the data, it
                     # should stay .tiff
        ),
        "wb")


def tile_writer(tile, fh):
    tile.copy(fh)
    return ImageFormat.TIFF


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

    image_stack = build_hybridization_stack(input_dir)
    image_stack_name = "hybridization.json"
    Writer.write_to_path(
        image_stack,
        os.path.join(output_dir, image_stack_name),
        pretty=True,
        tile_opener=tile_opener,
        tile_writer=tile_writer)

    codebook = [
        {'barcode': "AAGC", 'gene': "ACTB_human"},
        {'barcode': "AGGC", 'gene': "ACTB_mouse"},
    ]
    codebook_json_filename = "codebook.json"
    write_json(codebook, os.path.join(output_dir, codebook_json_filename))

    starfish_input = build_fov(input_dir, image_stack_name, codebook_json_filename, output_dir)
    starfish_input_name = "experiment.json"
    write_json(starfish_input, os.path.join(output_dir, starfish_input_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())
    parser.add_argument("--d", help="Download data", type=bool)

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir, args.d)
