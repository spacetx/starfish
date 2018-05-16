#!/usr/bin/env python

import argparse
import cProfile
import json
import os
import sys
from pstats import Stats

from .image import Indices
from .pipeline import registration
from .pipeline.gene_assignment import GeneAssignment
from starfish.pipeline.features.spots.detector import SpotFinder
from .pipeline.decoder import Decoder
from .util.argparse import FsExistsType


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--noop", help=argparse.SUPPRESS, dest="starfish_command", action="store_const", const=noop)

    subparsers = parser.add_subparsers(dest="starfish_command")

    if sys.version_info < (3, 0) and PROFILER_NOOP_ENVVAR in os.environ:
        noop_group = subparsers.add_parser("noop", add_help=False)
        noop_group.set_defaults(starfish_command=noop)

    registration.Registration.add_to_parser(subparsers)

    filter_group = subparsers.add_parser("filter")
    filter_group.add_argument("in_json", type=FsExistsType())
    filter_group.add_argument("out_dir", type=FsExistsType())
    filter_group.add_argument("--ds", default=15, type=int, help="Disk size")
    filter_group.set_defaults(starfish_command=filter)

    SpotFinder.add_to_parser(subparsers)

    segment_group = subparsers.add_parser("segment")
    segment_group.add_argument("in_json", type=FsExistsType())
    segment_group.add_argument("results_dir", type=FsExistsType())
    segment_group.add_argument("aux_image")
    segment_group.add_argument("--dt", default=.16, type=float, help="DAPI threshold")
    segment_group.add_argument("--st", default=.22, type=float, help="Input threshold")
    segment_group.add_argument("--md", default=57, type=int, help="Minimum distance between cells")
    segment_group.set_defaults(starfish_command=segment)

    GeneAssignment.add_to_parser(subparsers)
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
PROFILER_NOOP_ENVVAR = 'PROFILE_TEST'
"""If this environment variable is present, we create a no-op command for the purposes of testing the profiler."""


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


def filter(args, print_help=False):
    import numpy as np

    from .filters import white_top_hat
    from .io import Stack

    print('Filtering ...')
    print('Reading data')
    s = Stack()
    s.read(args.in_json)

    # filter dots
    print("Filtering dots ...")
    dots_filt = white_top_hat(s.aux_dict['dots'], args.ds)

    # create a 'stain' for segmentation
    # TODO: (ambrosejcarr) is this the appropriate way of dealing with Z in stain generation?
    stain = np.mean(s.max_proj(Indices.CH, Indices.Z), axis=0)
    stain = stain / stain.max()

    # filter raw images, for all hybs, channels
    for hyb in range(s.image.num_hybs):
        for ch in range(s.image.num_chs):
            for zlayer in range(s.image.num_zlayers):
                print("Filtering image: hyb={} ch={} zlayer={}...".format(hyb, ch, zlayer))
                indices = {Indices.HYB: hyb, Indices.CH: ch, Indices.Z: zlayer}
                data, axes = s.image.get_slice(indices=indices)
                assert len(axes) == 0
                result = white_top_hat(data, args.ds)
                s.image.set_slice(indices=indices, data=result)

    print("Writing results ...")
    s.set_aux('dots', dots_filt)
    s.set_aux('stain', stain)

    s.write(args.out_dir)


def segment(args, print_help=False):
    from .io import Stack
    from .munge import regions_to_geojson
    from .stats import label_to_regions
    from .watershedsegmenter import WatershedSegmenter

    s = Stack()
    s.read(args.in_json)

    # TODO make these parameterizable or determine whether they are useful or not
    size_lim = (10, 10000)
    disk_size_markers = None
    disk_size_mask = None

    seg = WatershedSegmenter(s.aux_dict['nuclei'], s.aux_dict[args.aux_image])
    cells_labels = seg.segment(args.dt, args.st, size_lim, disk_size_markers, disk_size_mask, args.md)

    regions = label_to_regions(cells_labels)
    geojson = regions_to_geojson(regions, use_hull=False)

    path = os.path.join(args.results_dir, 'regions.geojson')
    print("Writing | regions geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))


def show(args, print_help=False):
    import matplotlib.pyplot as plt
    from showit import tile

    from .io import Stack

    s = Stack()
    s.read(args.in_json)
    tile(s.squeeze(), size=args.sz, bar=True)
    plt.show()


def noop(args, print_help=False):
    pass
