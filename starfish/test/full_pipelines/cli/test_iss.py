import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from starfish.util import clock


def get_codebook(tempdir):
    json_filepath = os.path.join(tempdir, "formatted", "experiment.json")
    with open(json_filepath, "r") as fh:
        document = json.load(fh)
        return os.path.join(tempdir, "formatted", document['codebook'])


class TestWithIssData(unittest.TestCase):
    SUBDIRS = (
        "raw",
        "formatted",
        "registered",
        "filtered",
        "results",
    )

    STAGES = (
        [
            sys.executable,
            "examples/get_iss_data.py",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "raw"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted"),
            "--d", "1",
        ],
        [
            "starfish", "register",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted", "experiment.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered"),
            "fourier_shift",
            "--upsampling", "1000",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered", "experiment.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered"),
            "white_tophat",
            "--disk-size", "15",
        ],
        [
            "starfish", "detect_spots",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "experiment.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "gaussian_spot_detector",
            "--blobs-image-name", "dots",
            "--min-sigma", "4",
            "--max-sigma", "6",
            "--num-sigma", "20",
            "--threshold", "0.01",
        ],
        [
            "starfish", "segment",
            "-i", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "experiment.json"),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.geojson"),
            "watershed",
            "--dapi-threshold", ".16",
            "--input-threshold", ".22",
            "--min-distance", "57",
        ],
        [
            "starfish", "gene_assignment",
            "--coordinates-geojson",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.geojson"),
            "--spots-json", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "spots.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.json"),
            "point_in_poly",
        ],
        [
            "starfish", "decode",
            "-i", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "encoder_table.json"),
            "--codebook", lambda tempdir, *args, **kwargs: get_codebook(tempdir),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "decoded_table.json"),
            "iss",
        ],
    )

    def test_run_pipeline(self):
        tempdir = tempfile.mkdtemp()
        coverage_enabled = "STARFISH_COVERAGE" in os.environ

        def callback(interval):
            print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

        try:
            for subdir in TestWithIssData.SUBDIRS:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))
            for stage in TestWithIssData.STAGES:
                cmdline = [
                    element(tempdir=tempdir) if callable(element) else element
                    for element in stage
                ]
                if cmdline[0] == 'starfish' and coverage_enabled:
                    coverage_cmdline = [
                        "coverage", "run",
                        "-p",
                        "--source", "starfish",
                        "-m", "starfish",
                    ]
                    coverage_cmdline.extend(cmdline[1:])
                    cmdline = coverage_cmdline
                with clock.timeit(callback):
                    subprocess.check_call(cmdline)
            with open(os.path.join(tempdir, "results", "decoded_table.json")) as fh:
                results = json.load(fh)

            counts = collections.defaultdict(lambda: 0)
            for record in results:
                counts[record['barcode']] += 1
            tuples = [(count, barcode) for barcode, count in counts.items()]
            tuples.sort(reverse=True)
            self.assertEqual("AAGC", tuples[0][1])
            self.assertEqual("AGGC", tuples[1][1])
        finally:
            shutil.rmtree(tempdir)
