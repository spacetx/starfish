import os
import subprocess


def test_profiler():
    """Make sure that `starfish --profile works."""
    cmdline = [
        "starfish",
        "--profile",
        "version",
    ]
    env = os.environ.copy()
    subprocess.check_call(cmdline, env=env)
