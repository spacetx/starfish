#!/usr/bin/env python

import argparse
import cProfile
from pstats import Stats

from .pipeline.features.spots.decoder import Decoder
from .pipeline.features.spots.detector import SpotFinder
from .pipeline.filter import Filter
from .pipeline.target_assignment import TargetAssignment
from .pipeline.registration import Registration
from .pipeline.segmentation import Segmentation
from .util.argparse import FsExistsType


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--noop", help=argparse.SUPPRESS, dest="starfish_command", action="store_const", const=noop)

    subparsers = parser.add_subparsers(dest="starfish_command")

    Registration.add_to_parser(subparsers)
    Filter.add_to_parser(subparsers)
    SpotFinder.add_to_parser(subparsers)
    Segmentation.add_to_parser(subparsers)
    TargetAssignment.add_to_parser(subparsers)
    Decoder.add_to_parser(subparsers)

    show_group = subparsers.add_parser("show")
    show_group.add_argument("in_json", type=FsExistsType())
    show_group.add_argument("--sz", default=10, type=int, help="Figure size")
    show_group.set_defaults(starfish_command=show)

    return parser


PROFILER_KEY = "profiler"
"""This is the dictionary key we use to attach the profiler to pass to the resultcallback."""
PROFILER_LINES = 15
"""This is the number of profiling rows to dump when --profile is enabled."""


def starfish():
    parser = build_parser()
    args, argv = parser.parse_known_args()

    art = """
         _              __ _     _
        | |            / _(_)   | |
     ___| |_ __ _ _ __| |_ _ ___| |__
    / __| __/ _` | '__|  _| / __| '_  `
    \__ \ || (_| | |  | | | \__ \ | | |
    |___/\__\__,_|_|  |_| |_|___/_| |_|

    """
    print(art)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.starfish_command is None:
        parser.print_help()
        parser.exit(status=2)
    args.starfish_command(args, len(argv) != 0)

    if args.profile:
        stats = Stats(profiler)
        stats.sort_stats('tottime').print_stats(PROFILER_LINES)


def show(args, print_help=False):
    import matplotlib.pyplot as plt
    from showit import tile

    from .io import Stack

    s = Stack()
    s.read(args.in_json)
    tile(s.image.squeeze(), size=args.sz, bar=True)
    plt.show()


def noop(args, print_help=False):
    pass
