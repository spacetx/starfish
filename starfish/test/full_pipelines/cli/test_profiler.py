import os
import subprocess


def test_profiler():
    """Make sure that `starfish --profile --noop works."""
    cmdline = [
        "starfish",
        "--profile",
        "--noop",
    ]
    env = os.environ.copy()
    subprocess.check_call(cmdline, env=env)
