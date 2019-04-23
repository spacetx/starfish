import os
import subprocess
import sys
import tempfile
from pathlib import Path

from starfish.core.experiment.experiment import Experiment


def test_inplace():
    this = Path(__file__)
    inplace_script = this.parent / "inplace_script.py"

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        subprocess.check_call([sys.executable, os.fspath(inplace_script), os.fspath(tmpdir)])
        Experiment.from_json(os.fspath(tmpdir / "experiment.json"))
