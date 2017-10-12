#!/usr/bin/env python
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import json
from showit import tile

from .assign import assign
from .filters import white_top_hat
from .io import Stack
from .register import compute_shift, shift_im
from .spots.gaussian import GaussianSpotDetector
from .watershedsegmenter import WatershedSegmenter
from .stats import label_to_regions
from .munge import spots_to_geojson, regions_to_geojson

import pandas as pd


@click.group()
def starfish():
    art = """
         _              __ _     _
        | |            / _(_)   | |
     ___| |_ __ _ _ __| |_ _ ___| |__
    / __| __/ _` | '__|  _| / __| '_  `
    \__ \ || (_| | |  | | | \__ \ | | |
    |___/\__\__,_|_|  |_| |_|___/_| |_|

    """
    print(art)


@starfish.command()
@click.argument('in_json', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.option('--u', default=1, help='Amount of up-sampling', type=int)
def register(in_json, out_dir, u):
    print('Registering ...')
    s = Stack()
    s.read(in_json)

    mp = s.max_proj('ch')
    res = np.zeros(s.shape)

    for h in range(s.num_hybs):
        # compute shift between maximum projection (across channels) and dots, for each hyb round
        shift, error = compute_shift(mp[h, :, :], s.aux_dict['dots'], u)
        print("For hyb: {}, Shift: {}, Error: {}".format(h, shift, error))

        for c in range(s.num_chs):
            # apply shift to all channels and hyb rounds
            res[h, c, :] = shift_im(s.data[h, c, :], shift)

    s.set_stack(res)

    s.write(out_dir)


@starfish.command()
@click.argument('in_json', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.option('--ds', default=15, help='Disk size', type=int)
def filter(in_json, out_dir, ds):
    print('Filtering ...')
    print('Reading data')
    s = Stack()
    s.read(in_json)

    # filter raw images, for all hybs and channels
    stack_filt = []
    for im_num, im in enumerate(s.squeeze()):
        print("Filtering image: {}...".format(im_num))
        im_filt = white_top_hat(im, ds)
        stack_filt.append(im_filt)

    stack_filt = s.un_squeeze(stack_filt)

    # filter dots
    print("Filtering dots ...")
    dots_filt = white_top_hat(s.aux_dict['dots'], ds)

    print("Writing results ...")
    # create a 'stain' for segmentation
    stain = np.mean(s.max_proj('ch'), axis=0)
    stain = stain / stain.max()

    # update stack
    s.set_stack(stack_filt)
    s.set_aux('dots', dots_filt)
    s.set_aux('stain', stain)

    s.write(out_dir)


@starfish.command()
@click.argument('in_json', type=click.Path(exists=True))
@click.argument('results_dir', type=click.Path(exists=True))
@click.argument('aux_img', type=str)
@click.option('--min_sigma', default=4, help='Minimum spot size (in standard deviation)', type=int)
@click.option('--max_sigma', default=6, help='Maximum spot size (in standard deviation)', type=int)
@click.option('--num_sigma', default=20, help='Number of scales to try', type=int)
@click.option('--t', default=.01, help='Dots threshold', type=float)
@click.option('--show', default=False, help='Dots threshold', type=bool)
def detect_spots(in_json, results_dir, aux_img, min_sigma, max_sigma, num_sigma, t, show):
    print('Finding spots...')
    s = Stack()
    s.read(in_json)

    gsp = GaussianSpotDetector(s.squeeze(), s.aux_dict[aux_img])

    gsp.detect(min_sigma, max_sigma, num_sigma, t)
    if show:
        gsp.show(figsize=(10, 10))

    spots_viz = gsp.spots_df_viz
    spots_df_tidy = gsp.to_encoder_dataframe(tidy_flag=True, mapping=s.squeeze_map)

    geojson = spots_to_geojson(spots_viz)

    path = os.path.join(results_dir, 'spots.json')
    print("Writing | spots geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))

    path = os.path.join(results_dir, 'spots_geo.csv')
    print("Writing | spot_id | x | y | z | to: {}".format(path))
    spots_viz.to_csv(path, index=False)

    path = os.path.join(results_dir, 'encoder_table.csv')
    print("Writing | spot_id | hyb | ch | val | to: {}".format(path))
    spots_df_tidy.to_csv(path, index=False)


@starfish.command()
@click.argument('in_json', type=click.Path(exists=True))
@click.argument('results_dir', type=click.Path(exists=True))
@click.argument('aux_image')
@click.option('--dt', default=.16, help='DAPI threshold', type=float)
@click.option('--st', default=.22, help='Input threshold', type=float)
@click.option('--md', default=57, help='Minimum distance between cells', type=int)
def segment(in_json, results_dir, aux_image, dt, st, md):
    s = Stack()
    s.read(in_json)

    # TODO make these parameterizable or determine whether they are useful or not
    size_lim = (10, 10000)
    disk_size_markers = None
    disk_size_mask = None

    seg = WatershedSegmenter(s.aux_dict['dapi'], s.aux_dict[aux_image])
    cells_labels = seg.segment(dt, st, size_lim, disk_size_markers, disk_size_mask, md)

    r = label_to_regions(cells_labels)
    geojson = regions_to_geojson(r)

    path = os.path.join(results_dir, 'regions.json')
    print("Writing | regions geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))

    spots_geo = pd.read_csv(os.path.join(results_dir, 'spots_geo.csv'))
    # TODO only works in 3D
    points = spots_geo.loc[:, ['x', 'y']].values
    res = assign(cells_labels, points, use_hull=True)

    path = os.path.join(results_dir, 'regions.csv')
    print("Writing | cell_id | spot_id to: {}".format(path))
    res.to_csv(path, index=False)


@starfish.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--decoder_type', default='iss', help='Decoder type')
def decode(results_dir, decoder_type):
    if decoder_type == 'iss':
        from .decoders.iss import decode as dec
    else:
        raise ValueError('Decoder type: {} not supported'.format(decoder_type))

    encoder_table = pd.read_csv(os.path.join(results_dir, 'encoder_table.csv'))
    res = dec(encoder_table)
    path = os.path.join(results_dir, 'decoder_table.csv')
    print("Writing | spot_id | gene_id to: {}".format(path))
    res.to_csv(path, index=False)


@starfish.command()
@click.argument('in_json', type=click.Path(exists=True))
@click.option('--sz', default=10, help='Figure size', type=int)
def show(in_json, sz):
    s = Stack()
    s.read(in_json)
    tile(s.squeeze(), size=sz, bar=True)
    plt.show()
