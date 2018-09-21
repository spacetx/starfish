import os
import shutil
import unittest
from typing import List, Tuple

from starfish.intensity_table import IntensityTable
from starfish.util import exec


class CLITest(unittest.TestCase):
    """This is a base class for testing CLI methods. Each stage should correspond
    to a different pipeline step. Running the test will go through each stage and
    run the command line method with given arguments. The last stage should produce
    a file called results containing an IntensityTable. Each cli test should define
    it's own verify_results method.
    """
    # Since this is a base class we don't want to actually run it as a test.
    # For classes that inherit CLITest this will need to be set to True.
    __test__ = False

    SUBDIRS = Tuple[str]
    STAGES = Tuple[List]
    spots_file = "spots.nc"

    def verify_results(self, intensities):
        raise NotImplementedError()

    def test_run_pipline(self):
        tempdir = exec.stages(self.STAGES, self.SUBDIRS, keep_data=True)
        intensities = IntensityTable.load(os.path.join(tempdir, "results", self.spots_file))
        self.verify_results(intensities)

        if os.getenv("TEST_KEEP_DATA") is None:
            shutil.rmtree(tempdir)
