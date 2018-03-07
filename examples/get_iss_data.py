from __future__ import division

import argparse
import io
import os
import json
import zipfile

import requests
from skimage.io import imread, imsave

from starfish.util.argparse import FsExistsType


def download(input_dir, url):
    print("Downloading data ...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(input_dir)


def build_hybridization_stack(input_dir, output_dir):
    prefix = "fov_0"

    imgs = list()
    for hyb in range(1, 5):
        hyb_round = list()
        for ch in range(2, 6):
            hyb_round.append(imread(os.path.join(input_dir, str(hyb), "c{}.TIF".format(ch))))
        imgs.append(hyb_round)

    hybridization_stack = {
        'version': "0.0.0",
        'tiles': [],
    }

    for h, hyb in enumerate(imgs):
        for c, img in enumerate(hyb):
            fname = "{}_H_{}_C_{}.tiff".format(prefix, h, c)
            path = output_dir + fname
            d = {
                "coordinates":
                    {
                        "x": 0,
                        "y": 0,
                        "hyb": h,
                        "ch": c
                    },
                "file":
                    fname,
            }
            hybridization_stack["tiles"].append(d)
            imsave(path, img)

    if 2 < len(img.shape) > 3:
        raise ValueError("Images must be 2D or 3D. Found: {}".format(img.shape))

    hybridization_stack["legend"] = {
        "dimensions": ["x", "y", "hyb", "ch"],
        "default_tile_shape": img.shape,
        "default_tile_format": "TIFF",
    }

    return hybridization_stack


def build_fov(input_dir, hybridization_stack_name, output_dir):
    prefix = "fov_0"

    nuclei = imread(input_dir + "DO/c1.TIF")
    dots = imread(input_dir + "DO/c2.TIF")

    experiment = {
        'version': "0.0.0",
        'hybridization': hybridization_stack_name,
        'aux_img': {},
    }

    nuclei_fname = "{}_{}.tiff".format(prefix, "nuclei")
    imsave(output_dir + nuclei_fname, nuclei)
    experiment['aux_img']['nuclei'] = {
        'file': nuclei_fname,
        'tile_shape': nuclei.shape,
        'tile_format': "TIFF",
    }

    dots_fname = "{}_{}.tiff".format(prefix, "dots")
    imsave(output_dir + dots_fname, dots)
    experiment['aux_img']['dots'] = {
        'file': dots_fname,
        'tile_shape': dots.shape,
        'tile_format': "TIFF",
    }

    return experiment


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

    image_stack = build_hybridization_stack(input_dir, output_dir)
    image_stack_name = "hybridization.json"
    write_json(image_stack, os.path.join(output_dir, image_stack_name))

    starfish_input = build_fov(input_dir, image_stack_name, output_dir)
    starfish_input_name = "experiment.json"
    write_json(starfish_input, os.path.join(output_dir, starfish_input_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())
    parser.add_argument("--d", help="Download data", type=bool)

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir, args.d)
