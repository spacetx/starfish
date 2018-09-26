import argparse
import json
import pathlib

import requests

from starfish import Experiment
from starfish.util.argparse import FsExistsType


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_url")
    parser.add_argument("output_dir", type=FsExistsType())
    args = parser.parse_args()

    # save image stacks locally
    # exp = Experiment.from_json(str(pathlib.PurePosixPath(args.experiment_url, "experiment.json")))
    exp = Experiment.from_json(args.experiment_url + "experiment.json")

    for fov in exp.fovs():
        fov_dir = pathlib.Path(args.output_dir, fov.name)
        fov_dir.mkdir()
        fov.primary_image.write(str(fov_dir / "hybridization.json"))
        for image_type in fov.auxiliary_image_types:
            fov[image_type].write(str(fov_dir / f"{image_type}.json"))

    # get codebook from url and save locally to tmp dir
    codebook = requests.get(args.experiment_url + "codebook.json")
    # codebook = requests.get(str(pathlib.PurePosixPath(args.experiment_url, "codebook.json")))
    data = codebook.json()
    with open(args.output_dir + '/codebook.json', 'w') as f:
        json.dump(data, f)

    # TODO save exp.json
