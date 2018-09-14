import os
import shutil
import subprocess
import tempfile
import unittest
from typing import List, Tuple

from starfish.intensity_table import IntensityTable
from starfish.util import clock


class CLITest(unittest.TestCase):
    """This is a base class for testing CLI methods. Each stage should correspond to a different pipeline step.
    Running the test will go through each stage and run the command line method with given arguments.
    The last stage should produce a file called results containing an IntensityTable.
    Each cli test should define it's own verify_results method.
    """
    # Since this is a base class we don't want to actually run it as a test.
    # #For classes that inherit CLITest this will need to be set to True.
    __test__ = False

    SUBDIRS = Tuple[str]
    STAGES = Tuple[List]
    spots_file = "spots.nc"

    def verify_results(self, intensities):
        raise NotImplementedError()

    def test_run_pipline(self):
        tempdir = tempfile.mkdtemp()
        coverage_enabled = "STARFISH_COVERAGE" in os.environ
        print(tempdir)

        def callback(interval):
            print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

        try:
            for subdir in self.SUBDIRS:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))
            for i, stage in enumerate(self.STAGES):
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
                with clock.timeit(callback):
                    subprocess.check_call(cmdline)
            intensities = IntensityTable.load(os.path.join(tempdir, "results", self.spots_file))
            self.verify_results(intensities)

        finally:
            if os.getenv("TEST_KEEP_DATA") is None:
                shutil.rmtree(tempdir)
