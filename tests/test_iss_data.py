import collections
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # noqa
sys.path.insert(0, pkg_root)  # noqa


from starfish.util import clock


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
            "fourier_shift",
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
                if cmdline[0] == 'starfish':
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
            with open(os.path.join(tempdir, "results", "decoder_table.csv")) as fh:
                lines = fh.readlines()
                counts = collections.defaultdict(lambda: 0)
                for line in lines[1:]:
                    line_barcode = line.split(",")[0]
                    counts[line_barcode] += 1

                tuples = [(count, barcode) for barcode, count in counts.items()]
                tuples.sort(reverse=True)
                self.assertEqual("AAGC", tuples[0][1])
                self.assertEqual("AGGC", tuples[1][1])
        finally:
            shutil.rmtree(tempdir)
