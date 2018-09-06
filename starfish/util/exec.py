import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Sequence

import jsonpath_rw
import numpy as np
import pandas as pd

from starfish.intensity_table import IntensityTable
from starfish.types import Features
from starfish.util import clock


def stages(commands, subdirs=None, keep_data=False):
    """
    Execute a list of commands in a temporary directory
    cleaning them up unless otherwise requested.
    """
    tempdir = tempfile.mkdtemp()
    coverage_enabled = "STARFISH_COVERAGE" in os.environ

    def callback(interval):
        print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

    try:

        if subdirs:
            for subdir in subdirs:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))

        for stage in commands:
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
        if not keep_data:
            shutil.rmtree(tempdir)
