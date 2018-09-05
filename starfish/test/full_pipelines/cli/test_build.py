import os
import shutil
import subprocess
import tempfile
import unittest


from starfish.util import clock


class TestWithBuildData(unittest.TestCase):

    STAGES = (
        [
            "starfish", "build",
            "--fov-count=2", '--hybridization-dimensions={"z": 3}',
            lambda tempdir: tempdir
        ],
        [
            "validate-sptx", "--experiment-json",
            lambda tempdir: os.sep.join([tempdir, "experiment.json"])
        ],
    )

    def test_run_build(self):
        tempdir = tempfile.mkdtemp()
        coverage_enabled = "STARFISH_COVERAGE" in os.environ

        def callback(interval):
            print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

        try:
            # TODO: duplicated, time to refactor
            for stage in TestWithBuildData.STAGES:
                cmdline = [
                    element(tempdir=tempdir) if callable(element) else element
                    for element in stage
                ]
                if cmdline[0] == "starfish" and coverage_enabled:
                    coverage_cmdline = [
                        "coverage", "run",
                        "-p",
                        "--source", "starfish",
                        "-m", "starfish",
                    ]
                    coverage_cmdline.extend(cmdline[1:])
                    cmdline = coverage_cmdline
                elif cmdline[0] == "validate-sptx" and coverage_enabled:
                    coverage_cmdline = [
                        "coverage", "run",
                        "-p",
                        "--source", "validate_sptx",
                        "-m", "validate_sptx",
                    ]
                    coverage_cmdline.extend(cmdline[1:])
                    cmdline = coverage_cmdline
                with clock.timeit(callback):
                    subprocess.check_call(cmdline)

        finally:
            if os.getenv("TEST_BUILD_KEEP_DATA") is None:
                shutil.rmtree(tempdir)
