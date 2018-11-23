#!/usr/bin/env python
import cProfile
import json
import sys
from pstats import Stats

import click

from sptx_format.cli import validate as validate_cli
from starfish.experiment.builder.cli import build as build_cli
from starfish.image import (
    Filter,
    Registration,
    Segmentation,
)
from starfish.spots import (
    Decoder,
    SpotFinder,
    TargetAssignment,
)
from starfish.util.config import Config


PROFILER_KEY = "profiler"
"""This is the dictionary key we use to attach the profiler to pass to the resultcallback."""
PROFILER_LINES = 15
"""This is the number of profiling rows to dump when --profile is enabled."""


@click.group()
@click.option("--profile", is_flag=True)
@click.option("--quiet", is_flag=True)
@click.option("--record", type=click.Path(writable=True, allow_dash=True))
@click.pass_context
def starfish(ctx, profile, quiet=False, record=None):
    config = Config()
    art = """
         _              __ _     _
        | |            / _(_)   | |
     ___| |_ __ _ _ __| |_ _ ___| |__
    / __| __/ _` | '__|  _| / __| '_  `
    \__ \ || (_| | |  | | | \__ \ | | |
    |___/\__\__,_|_|  |_| |_|___/_| |_|

    """  # noqa
    print_art = True
    sub = ctx.command.get_command(ctx, ctx.invoked_subcommand)
    if config.lookup(["cli", "quiet"], False) or quiet:
        print_art = False
    elif hasattr(sub, "no_art"):
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

    ctx.obj = {"record": None}
    if record is None:
        record = config.lookup(["cli", "record"], None)
    if record:
        v = []
        ctx.obj["record"] = v

        def write_record():
            if "-" == record:
                json.dump(v, sys.stdout)
            else:
                with open(record, "a") as o:
                    json.dump(v, o)

        ctx.call_on_close(write_record)


@starfish.command()
def version():
    import pkg_resources
    version = pkg_resources.require("starfish")[0].version
    print(version)
version.no_art = True  # type: ignore


# Pipelines
starfish.add_command(Registration._cli)  # type: ignore
starfish.add_command(Filter._cli)  # type: ignore
starfish.add_command(SpotFinder._cli)  # type: ignore
starfish.add_command(Segmentation._cli)  # type: ignore
starfish.add_command(TargetAssignment._cli)  # type: ignore
starfish.add_command(Decoder._cli)  # type: ignore

# Other
starfish.add_command(build_cli)  # type: ignore
starfish.add_command(validate_cli)  # type: ignore
