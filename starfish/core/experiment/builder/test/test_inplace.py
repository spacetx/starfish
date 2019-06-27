import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from starfish.core.experiment.experiment import Experiment, FieldOfView


def test_inplace(tmpdir):
    this = Path(__file__)
    inplace_script = this.parent / "inplace_script.py"

    tmpdir_path = Path(tmpdir)

    subprocess.check_call([sys.executable, os.fspath(inplace_script), os.fspath(tmpdir_path)])

    # load up the experiment, and select an image.  Ensure that it has non-zero data.  This is to
    # verify that we are sourcing the data from the tiles that were already on-disk, and not the
    # artificially zero'ed tiles that we feed the experiment builder.
    experiment = Experiment.from_json(os.fspath(tmpdir_path / "experiment.json"))
    primary_image = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
    assert not np.allclose(primary_image.xarray, 0)
