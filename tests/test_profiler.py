import os
import subprocess
import sys
import unittest


pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # noqa
sys.path.insert(0, pkg_root)  # noqa


from starfish.starfish import PROFILER_NOOP_ENVVAR


class TestProfiler(unittest.TestCase):

    def test_profiler(self):
        """Make sure that `starfish --profile --noop works."""
        cmdline = [
            "starfish",
            "--profile",
            "--noop",
        ]
        env = os.environ.copy()
        if sys.version_info < (3, 0):
            # this is a hideous hack because argparse is not consistent between 2.7 and 3.x about whether the subcommand
            # is required.  this entire hack can be dropped if we deprecate python2.7.
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
