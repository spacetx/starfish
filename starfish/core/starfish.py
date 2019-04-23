#!/usr/bin/env python
import cProfile
import subprocess
import sys
from pstats import Stats

import pkg_resources

from starfish.core.experiment.builder.cli import build as build_cli
from starfish.core.image import (
    ApplyTransform,
    Filter,
    LearnTransform,
    Segment,
)
from starfish.core.spacetx_format.cli import validate as validate_cli
from starfish.core.spots import (
    AssignTargets,
    Decode,
    DetectPixels,
    DetectSpots,
)
from starfish.core.util import click


PROFILER_KEY = "profiler"
"""This is the dictionary key we use to attach the profiler to pass to the resultcallback."""
PROFILER_LINES = 15
"""This is the number of profiling rows to dump when --profile is enabled."""


def art_string():
    return r"""
         _              __ _     _
        | |            / _(_)   | |
     ___| |_ __ _ _ __| |_ _ ___| |__
    / __| __/ _` | '__|  _| / __| '_  `
    \__ \ || (_| | |  | | | \__ \ | | |
    |___/\__\__,_|_|  |_| |_|___/_| |_|

    """

@click.group()
@click.option("--profile", is_flag=True)
@click.pass_context
def starfish(ctx, profile):
    """
    standardized analysis pipeline for image-based transcriptomics
    see: https://spacetx-starfish.readthedocs.io for more information.
    """
    art = art_string()
    print_art = True
    sub = ctx.command.get_command(ctx, ctx.invoked_subcommand)
    if hasattr(sub, "no_art"):
        print_art = not sub.no_art
    if print_art:
        print(art)
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

        def print_profile():
            stats = Stats(profiler)
            stats.sort_stats('tottime').print_stats(PROFILER_LINES)

        ctx.call_on_close(print_profile)


@starfish.command()
def version():
    from starfish import __version__
    print(__version__)
version.no_art = True  # type: ignore


@starfish.group()
def util():
    """
    house-keeping commands for the starfish library
    """
    pass

@util.command()
def install_strict_dependencies():
    """
    warning! updates different packages in your local installation
    """
    strict_requirements_file = pkg_resources.resource_filename(
        "starfish", "REQUIREMENTS-STRICT.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", strict_requirements_file
    ])


# Pipelines
starfish.add_command(LearnTransform._cli)  # type: ignore
starfish.add_command(ApplyTransform._cli)  # type: ignore
starfish.add_command(Filter._cli)  # type: ignore
starfish.add_command(DetectPixels._cli)
starfish.add_command(DetectSpots._cli)  # type: ignore
starfish.add_command(Segment._cli)  # type: ignore
starfish.add_command(AssignTargets._cli)  # type: ignore
starfish.add_command(Decode._cli)  # type: ignore

# Other
starfish.add_command(build_cli)  # type: ignore
starfish.add_command(validate_cli)  # type: ignore
