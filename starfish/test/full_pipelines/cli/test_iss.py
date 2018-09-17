import json
import os
import sys
from typing import Sequence

import jsonpath_rw
import numpy as np
import pandas as pd

from starfish.test.full_pipelines.cli._base_cli_test import CLITest
from starfish.types import Features


def get_jsonpath_from_file(json_filepath_components: Sequence[str], jsonpath: str):
    """
    Given a series of filepath components, join them to find a file <FILE>.  Open that file, and
    locate a specific value in the json structure <PATH>.  Join the directory path of <FILE> and
    <PATH> and return that.

    For example, if json_filepath_components is ["/tmp", "formatted", "experiment.json"] and
    jsonpath is "$['hybridization']", this method will open /tmp/formatted/experiment.json, decode
    that as a json document, and locate the value of the key 'hybridization'.  It will return
    /tmp/formatted/XXX, where XXX is the value of the key.
    """
    json_filepath = os.path.join(*json_filepath_components)
    dirname = os.path.dirname(json_filepath)
    with open(json_filepath, "r") as fh:
        document = json.load(fh)
        return os.path.join(dirname, jsonpath_rw.parse(jsonpath).find(document)[0].value)


class TestWithIssData(CLITest):
    __test__ = True

    spots_file = "decoded-spots.nc"

    SUBDIRS = (
        "raw",
        "formatted",
        "registered",
        "filtered",
        "results",
    )

    STAGES = (
        # [
        #     sys.executable,
        #     "examples/get_iss_data.py",
        #     lambda tempdir, *args, **kwargs: os.path.join(tempdir, "raw"),
        #     lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted"),
        #     "--d", "1",
        # ],
        # TODO make this work
        [
            sys.executable,
            "examples/get_cli_test_data.py",
            "https://dmf0bdeheu4zf.cloudfront.net/20180828/ISS-TEST/test-iss-data.zip",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted")
        ],
        [
            "starfish", "registration",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [
                    tempdir,
                    get_jsonpath_from_file(
                        [tempdir, "formatted", "experiment.json"],
                        "$['primary_images']",
                    ),
                ],
                "$['contents']['fov_000']"
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered", "hybridization.json"),
            "FourierShiftRegistration",
            "--reference-stack", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [
                    tempdir,
                    get_jsonpath_from_file(
                        [tempdir, "formatted", "experiment.json"],
                        "$['auxiliary_images']['dots']",
                    ),
                ],
                "$['contents']['fov_000']"
            ),
            "--upsampling", "1000",
        ],
        [
            "starfish", "validate", "--experiment-json",
            lambda tempdir: os.sep.join([tempdir, "formatted", "experiment.json"])
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "hybridization.json"),
            "WhiteTophat",
            "--masking-radius", "15",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [
                    tempdir,
                    get_jsonpath_from_file(
                        [tempdir, "formatted", "experiment.json"],
                        "$['auxiliary_images']['nuclei']",
                    ),
                ],
                "$['contents']['fov_000']"
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "nuclei.json"),
            "WhiteTophat",
            "--masking-radius", "15",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [
                    tempdir,
                    get_jsonpath_from_file(
                        [tempdir, "formatted", "experiment.json"],
                        "$['auxiliary_images']['dots']",
                    ),
                ],
                "$['contents']['fov_000']"
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "dots.json"),
            "WhiteTophat",
            "--masking-radius", "15",
        ],
        [
            "starfish", "detect_spots",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "spots.nc"),
            "GaussianSpotDetector",
            "--blobs-stack", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "dots.json"),
            "--min-sigma", "4",
            "--max-sigma", "6",
            "--num-sigma", "20",
            "--threshold", "0.01",
        ],
        [
            "starfish", "segment",
            "--hybridization-stack", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "hybridization.json"),
            "--nuclei-stack", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "nuclei.json"),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "regions.geojson"),
            "Watershed",
            "--dapi-threshold", ".16",
            "--input-threshold", ".22",
            "--min-distance", "57",
        ],
        [
            "starfish", "target_assignment",
            "--coordinates-geojson",
            lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "regions.geojson"),
            "--intensities", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "spots.nc"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "targeted-spots.nc"),
            "PointInPoly2D",
        ],
        [
            "starfish", "decode",
            "-i", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "targeted-spots.nc"),
            "--codebook", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['codebook']",
            ),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "results", "decoded-spots.nc"),
            "PerRoundMaxChannelDecoder",
        ],
    )

    def verify_results(self, intensities):
        # TODO make this test stronger
        genes, counts = np.unique(
            intensities.coords[Features.TARGET], return_counts=True)
        gene_counts = pd.Series(counts, genes)
        assert gene_counts['ACTB_human'] > gene_counts['ACTB_mouse']
