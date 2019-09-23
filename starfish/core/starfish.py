#!/usr/bin/env python
import subprocess
import sys

import pkg_resources

from starfish.core.spacetx_format.cli import validate as validate_cli
from starfish.core.util import click


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
@click.pass_context
def starfish(ctx):
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


# Other
starfish.add_command(validate_cli)  # type: ignore
