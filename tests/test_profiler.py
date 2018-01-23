import os
import subprocess
import sys
import unittest


pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # noqa
sys.path.insert(0, pkg_root)  # noqa


from starfish.starfish import PROFILER_NOOP_ENVVAR


class TestProfiler(unittest.TestCase):

    def test_profiler(self):
        """Make sure that `starfish --profile noop works."""
        cmdline = [
            "starfish",
            "--profile",
            "noop",
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
        env = os.environ.copy()
        env[PROFILER_NOOP_ENVVAR] = ""
        subprocess.check_call(cmdline, env=env)

    def test_noop_hiding(self):
        """
        Ensure we do not expose the noop command unless the magic environment flag is present.  This is to avoid
        confusion from the end users.
        """
        cmdline = [
            "starfish",
            "noop",
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

        env = os.environ.copy()
        env[PROFILER_NOOP_ENVVAR] = ""
        subprocess.check_call(cmdline, env=env)

        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call(cmdline)
