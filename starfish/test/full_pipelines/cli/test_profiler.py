import os
import subprocess
import sys

from starfish.starfish import PROFILER_NOOP_ENVVAR


def test_profiler():
    """Make sure that `starfish --profile --noop works."""
    cmdline = [
        "starfish",
        "--profile",
        "--noop",
    ]
    env = os.environ.copy()
    if sys.version_info < (3, 0):
        # this is a hideous hack because argparse is not consistent between 2.7 and 3.x about whether the subcommand
        # is required.  TODO ttung: this entire hack can be dropped if we deprecate python2.7.
        cmdline.append("noop")
        env[PROFILER_NOOP_ENVVAR] = ""
    if cmdline[0] == 'starfish':
        coverage_cmdline = [
            "coverage", "run",
            "-p",
            "--source", "starfish",
            "-m", "starfish",
        ]
        coverage_cmdline.extend(cmdline[1:])
        cmdline = coverage_cmdline
    subprocess.check_call(cmdline, env=env)
