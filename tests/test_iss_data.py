import os
import shutil
import subprocess
import sys
import tempfile
import unittest


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
            "python",
            "examples/get_iss_data.py",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "raw"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted"),
            "--d", "1",
        ],
        [
            "starfish", "register",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted", "org.json"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered"),
            "--u", "1000",
        ],
        [
            "starfish", "filter",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered", "org.json"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered"),
            "--ds", "15",
        ],
        [
            "starfish", "detect_spots",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "org.json"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "dots",
            "--min_sigma", "4",
            "--max_sigma", "6",
            "--num_sigma", "20",
            "--t", "0.01",
        ],
        [
            "starfish", "segment",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "org.json"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "stain",
            "--dt", ".16",
            "--st", ".22",
            "--md", "57",
        ],
        [
            "starfish", "decode",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "--decoder_type", "iss",
        ],
    )

    def test_run_pipeline(self):
        tempdir = tempfile.mkdtemp()

        try:
            for subdir in TestWithIssData.SUBDIRS:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))
            for stage in TestWithIssData.STAGES:
                cmdline = [
                    element(tempdir=tempdir) if callable(element) else element
                    for element in stage
                ]
                subprocess.check_call(cmdline)
        finally:
            shutil.rmtree(tempdir)
