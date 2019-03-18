#!/usr/bin/env python

from copy import deepcopy
from collections import defaultdict
import glob
import hashlib
import os
import json
import re

import click
import numpy as np
from skimage.io import imsave, imread

# note this reads the data twice because I forgot about the need to
# correct the hashes
@click.command()
@click.argument("directory", type=str)
def fix_hashes(directory):
    # note that this script leverages the fact that the files are originally named .tiff but are
    # in fact .npy. This means that np.load will correctly identify the .npy data, while
    # skimage.io.imsave will correctly infer from the same name that we want TIFF.
    # thus, this script will overwrite files with the correct data type without
    # any name changes.

    # write correct data; read # 1
    # files = glob.glob(os.path.join(directory, "*.tiff"))
    # for f in files:
    #     data = np.load(f)
    #     imsave(f, data)

    # upgrade the json hashes; read # 2
    for json_to_upgrade in ("primary_fixed_FOV_000.json", "nissl_fixed_FOV_000.json"):

        json_fname = os.path.join(directory, json_to_upgrade)
        with open(json_fname, "r") as f:
            json_data = json.load(f)
        upgraded_json = deepcopy(json_data)

        for i, tile in enumerate(json_data["tiles"]):
            filename = tile["file"]

            # update hashes
            m = hashlib.sha256()
            with open(filename, 'rb') as fin:
                m.update(fin.read())
            hexdigest = m.hexdigest()

            upgraded_json["tiles"][i]["sha256"] = hexdigest
            del upgraded_json["tiles"][i]["tile_format"]

        with open(json_fname, 'w') as fout:
            json.dump(upgraded_json, fout)

@click.command()
@click.argument("directory", type=str)
def split_fovs(directory):

    for json_to_upgrade in ("primary_fixed_FOV_000.json", "nissl_fixed_FOV_000.json"):

        json_fname = os.path.join(directory, json_to_upgrade)
        with open(json_fname, "r") as f:
            json_data = json.load(f)

        jsons = defaultdict(lambda: defaultdict(list))

        # going to write new jsons
        for i, tile in enumerate(json_data["tiles"]):
            filename = tile["file"]
            pattern = re.compile(r"seq\dalignedfixedT\d{5}C\d{2}Z\d{3}X(\d)Y(\d)")
            mo = re.match(pattern, filename)
            x, y = (int(i) - 1 for i in mo.groups())

            # remove the extras, it was formatted wrong
            del tile["extras"]

            jsons[x, y]["tiles"].append(tile)

        top_level_data = deepcopy(json_data)
        del top_level_data["tiles"]
        for k, v in jsons.items():
            jsons[k] = {**v, **top_level_data}

        stem, *_ = json_to_upgrade.split("_")

        for i, (k, d) in enumerate(jsons.items()):
            with open(f"{stem}_fov_{i:03d}.json", "w") as fout:
                json.dump(d, fout)

@click.command()
@click.argument("directory", type=str)
@click.argument("stem", type=str)
def make_manifest(directory, stem):

    data = {"version": "0.0.0", "contents": {}}

    target = os.path.join(directory, stem + "_fov*")
    files = glob.glob(target)
    for f in files:
        name = os.path.basename(f)
        base, ext = os.path.splitext(name)
        *_, fov = base.split("_")
        data["contents"][f"fov_{fov}"] = name

    with open(f"{stem}.json", "w") as fout:
        json.dump(data, fout)


@click.command()
@click.argument("manifest", type=str)
def fix_sizes(manifest):
    with open(manifest) as f:
        manifest_data = json.load(f)

    for k, fov_filename in manifest_data["contents"].items():
        with open(fov_filename) as f:
            fov_data = json.load(f)

            for tile in fov_data['tiles']:
                data = imread(tile['file'])

                # Sigh. Gotta crop everything.
                data = data[:1193, :913]
                shape = data.shape
                tile["tile_shape"] = shape
                imsave(tile['file'], data)

                # ... and update hashes
                m = hashlib.sha256()
                with open(tile['file'], 'rb') as fin:
                    m.update(fin.read())
                hexdigest = m.hexdigest()
                tile['sha256'] = hexdigest

                # save the data

        with open(fov_filename, "w") as f:
            json.dump(fov_data, f)

@click.command()
@click.argument("manifest", type=str)
def fix_indices(manifest):
    with open(manifest) as f:
        manifest_data = json.load(f)

    for k, fov_filename in manifest_data["contents"].items():
        with open(fov_filename) as f:
            fov_data = json.load(f)

        # fov_data["shape"]["c"] = 1

        # for tile in fov_data["tiles"]:
            # tile["indices"]["c"] = 0

        del fov_data["default_tile_shape"]

        with open(fov_filename, "w") as f:
            json.dump(fov_data, f)


if __name__ == "__main__":
    fix_indices()
