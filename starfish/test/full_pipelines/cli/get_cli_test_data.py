import argparse
import json
import pathlib
import posixpath

import requests

from starfish import Experiment, FieldOfView
from starfish.util.argparse import FsExistsType


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_url")
    parser.add_argument("output_dir", type=FsExistsType())
    args = parser.parse_args()

    # save image stacks locally
    exp = Experiment.from_json(posixpath.join(args.experiment_url, "experiment.json"))

    for fov in exp.fovs():
        fov_dir = pathlib.Path(args.output_dir, fov.name)
        fov_dir.mkdir()
        fov[FieldOfView.PRIMARY_IMAGES].export(str(fov_dir / "primary_images.json"))
        for image_type in fov.image_types:
            if image_type == FieldOfView.PRIMARY_IMAGES:
                continue
            fov[image_type].export(str(fov_dir / f"{image_type}.json"))

    # get codebook from url and save locally to tmp dir
    codebook = requests.get(posixpath.join(args.experiment_url, "codebook.json"))
    data = codebook.json()
    with open(pathlib.Path(args.output_dir, 'codebook.json'), 'w') as f:
        json.dump(data, f)

    # get json files from url and save locally to tmp dir
    for filename in ("experiment.json", "hybridization.json", "nuclei.json", "dots.json"):
        obj = requests.get(posixpath.join(args.experiment_url, filename))
        data = obj.json()
        with open(pathlib.Path(args.output_dir, filename), 'w') as f:
            json.dump(data, f)
