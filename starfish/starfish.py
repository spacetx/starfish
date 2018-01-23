#!/usr/bin/env python

import argparse
import cProfile
import json
import os
import sys

try:
    from pstats import Stats  # python 3.x
except ImportError:
    from profile import Stats  # python 2.x


from .util.argparse import FsExistsType
from . import registration


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

    detect_spots_group = subparsers.add_parser("detect_spots")
    detect_spots_group.add_argument("in_json", type=FsExistsType())
    detect_spots_group.add_argument("results_dir", type=FsExistsType())
    detect_spots_group.add_argument("aux_image")
    detect_spots_group.add_argument(
        "--min_sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
    detect_spots_group.add_argument(
        "--max_sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
    detect_spots_group.add_argument("--num_sigma", default=20, type=int, help="Number of scales to try")
    detect_spots_group.add_argument("--t", default=.01, type=float, help="Dots threshold")
    detect_spots_group.add_argument("--show", default=False, type=bool, help="Dots threshold")
    detect_spots_group.set_defaults(starfish_command=detect_spots)

    segment_group = subparsers.add_parser("segment")
    segment_group.add_argument("in_json", type=FsExistsType())
    segment_group.add_argument("results_dir", type=FsExistsType())
    segment_group.add_argument("aux_image")
    segment_group.add_argument("--dt", default=.16, type=float, help="DAPI threshold")
    segment_group.add_argument("--st", default=.22, type=float, help="Input threshold")
    segment_group.add_argument("--md", default=57, type=int, help="Minimum distance between cells")
    segment_group.set_defaults(starfish_command=segment)

    decode_group = subparsers.add_parser("decode")
    decode_group.add_argument("results_dir", type=FsExistsType())
    decode_group.add_argument("--decoder_type", default="iss", help="Decoder type")
    decode_group.set_defaults(starfish_command=decode)

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
    args = parser.parse_args()

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
    args.starfish_command(args)

    if args.profile:
        stats = Stats(profiler)
        stats.sort_stats('tottime').print_stats(PROFILER_LINES)


def filter(args):
    import numpy as np

    from .filters import white_top_hat
    from .io import Stack

    print('Filtering ...')
    print('Reading data')
    s = Stack()
    s.read(args.in_json)

    # filter raw images, for all hybs and channels
    stack_filt = []
    for im_num, im in enumerate(s.squeeze()):
        print("Filtering image: {}...".format(im_num))
        im_filt = white_top_hat(im, args.ds)
        stack_filt.append(im_filt)

    stack_filt = s.un_squeeze(stack_filt)

    # filter dots
    print("Filtering dots ...")
    dots_filt = white_top_hat(s.aux_dict['dots'], args.ds)

    print("Writing results ...")
    # create a 'stain' for segmentation
    stain = np.mean(s.max_proj('ch'), axis=0)
    stain = stain / stain.max()

    # update stack
    s.set_stack(stack_filt)
    s.set_aux('dots', dots_filt)
    s.set_aux('stain', stain)

    s.write(args.out_dir)


def detect_spots(args):
    from .io import Stack
    from .munge import spots_to_geojson
    from .spots.gaussian import GaussianSpotDetector

    print('Finding spots...')
    s = Stack()
    s.read(args.in_json)

    # create 'encoder table' standard (tidy) file format.
    gsp = GaussianSpotDetector(s)
    spots_df_tidy = gsp.detect(
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        num_sigma=args.num_sigma,
        threshold=args.t,
        blobs=args.aux_image,
        measurement_type='max',
        bit_map_flag=False
    )

    if args.show:
        gsp.show(figsize=(10, 10))

    spots_viz = gsp.spots_df_viz
    geojson = spots_to_geojson(spots_viz)

    path = os.path.join(args.results_dir, 'spots.json')
    print("Writing | spots geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))

    path = os.path.join(args.results_dir, 'spots_geo.csv')
    print("Writing | spot_id | x | y | z | to: {}".format(path))
    spots_viz.to_csv(path, index=False)

    path = os.path.join(args.results_dir, 'encoder_table.csv')
    print("Writing | spot_id | hyb | ch | val | to: {}".format(path))
    spots_df_tidy.to_csv(path, index=False)


def segment(args):
    import pandas as pd

    from .assign import assign
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

    seg = WatershedSegmenter(s.aux_dict['dapi'], s.aux_dict[args.aux_image])
    cells_labels = seg.segment(args.dt, args.st, size_lim, disk_size_markers, disk_size_mask, args.md)

    r = label_to_regions(cells_labels)
    geojson = regions_to_geojson(r)

    path = os.path.join(args.results_dir, 'regions.json')
    print("Writing | regions geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))

    spots_geo = pd.read_csv(os.path.join(args.results_dir, 'spots_geo.csv'))
    # TODO only works in 3D
    points = spots_geo.loc[:, ['x', 'y']].values
    res = assign(cells_labels, points, use_hull=True)

    path = os.path.join(args.results_dir, 'regions.csv')
    print("Writing | cell_id | spot_id to: {}".format(path))
    res.to_csv(path, index=False)


def decode(args):
    import pandas as pd

    encoder_table = pd.read_csv(os.path.join(args.results_dir, 'encoder_table.csv'))
    # TODO this should be loaded from disk
    d = {'barcode': ['AAGC', 'AGGC'], 'gene': ['ACTB_human', 'ACTB_mouse']}
    codebook = pd.DataFrame(d)
    if args.decoder_type == 'iss':
        from .decoders.iss import IssDecoder
        decoder = IssDecoder(codebook, letters=['T', 'G', 'C', 'A'])
    else:
        raise ValueError('Decoder type: {} not supported'.format(args.decoder_type))

    res = decoder.decode(encoder_table)
    path = os.path.join(args.results_dir, 'decoder_table.csv')
    print("Writing | spot_id | gene_id to: {}".format(path))
    res.to_csv(path, index=False)


def show(args):
    import matplotlib.pyplot as plt
    from showit import tile

    from .io import Stack

    s = Stack()
    s.read(args.in_json)
    tile(s.squeeze(), size=args.sz, bar=True)
    plt.show()


def noop(args):
    pass
