import os
import shutil

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.util import exec


class CLITest:
    """This is a base class for testing CLI methods. Each stage should correspond
    to a different pipeline step. Running the test will go through each stage and
    run the command line method with given arguments. The last stage should produce
    a file called results containing an IntensityTable. Each cli test should define
    it's own verify_results method.
    """
    @property
    def subdirs(self):
        raise NotImplementedError()

    @property
    def stages(self):
        raise NotImplementedError()

    @property
    def spots_file(self):
        return "spots.nc"

    def verify_results(self, intensities: IntensityTable):
        raise NotImplementedError()

    def test_run_pipline(self):
        tempdir = exec.stages(self.stages, self.subdirs, keep_data=True)
        intensities = IntensityTable.open_netcdf(os.path.join(tempdir, "results", self.spots_file))
        self.verify_results(intensities)

        if os.getenv("TEST_KEEP_DATA") is None:
            shutil.rmtree(tempdir)
