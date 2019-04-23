import os
import subprocess
import tempfile
from typing import Callable, Sequence, Union

from starfish.core.util import clock


def stages(commands: Sequence[Sequence[Union[str, Callable]]],
           subdirs: Sequence[str]=None,

           keep_data: bool=False) -> str:
    """
    Execute a list of commands in a temporary directory
    cleaning them up unless otherwise requested.

    Parameters
    ----------
    commands : Sequence[Sequence[Union[str, Callable]]]
        A collection of tuples of commands composed either of
        str elements or callable objects which will be invoked
        with the keyword argument "tempdir". The resulting list
        of strings will be passed to subprocess.check_call.
    subdirs : Sequence[str]
        A collection of paths which should be created as subdirectories
        within the temporary directory used by this invocation.
    keep_data : bool
        If not true, shutil.rmtree will be called on the temporary
        directory used by this invocation.
    STARFISH_COVERAGE :
        This parameter is read from the environment. If set, then command lists will have
        `coverage run ...` options prepended before execution.

    Return
    ------
    str :
        Path to the temporary directory used by this invocation.

    """

    if keep_data:
        tempobj = None
        tempdir = tempfile.mkdtemp()
    else:
        tempobj = tempfile.TemporaryDirectory()
        tempdir = tempobj.name

    def callback(interval):
        print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

    try:

        if subdirs:
            for subdir in subdirs:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))

        for stage in commands:
            cmdline = prepare_stage(stage, tempdir)
            with clock.timeit(callback):
                subprocess.check_call(cmdline)

        return tempdir

    finally:
        if tempobj:
            tempobj.cleanup()
        else:
            print("Temporary files kept under:")
            print(tempdir)


def prepare_stage(stage: Sequence[Union[str, Callable]],
                  tempdir: str) -> Sequence[str]:
    """
    Loop through elements of stage, building them into a commandline.
    If an element is a callable, it will be invoked with the "tempdir"
    keyword.

    Parameters
    ----------
    stage: Sequence[Union[str, Callable]]
        A collection of commands composed either of
        str elements or callable objects which will be invoked
        with the keyword argument "tempdir". The resulting list
        of strings will be passed to subprocess.check_call.
    tempdir: str
        Temporary directory that will be used by the invoking method.
    STARFISH_COVERAGE :
        This parameter is read from the environment. If set, then command lists will have
        `coverage run ...` options prepended before execution.

    Return
    ------
    Sequence of strings for passing to subprocess.check_call

    """
    coverage_enabled = "STARFISH_COVERAGE" in os.environ
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
    return cmdline
