from __future__ import division

import numpy as np
import pandas as pd
from skimage import io
import click
import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/dganguli/src/starfish/")

from starfish.filters import white_top_hat
from starfish.munge import list_to_stack, stack_to_list, max_proj, scale
from starfish.register import compute_shift, shift_im
from starfish.spots import BinarySpotDetector
from starfish.stats import im_describe
from starfish.watershedsegmenter import WatershedSegmenter


def load_data(base_path='/Users/dganguli/Downloads/ExampleInSituSequencing/'):
    def load_tiff_stack(path):
        stack = io.imread(path)
        return stack

    def load_hyb_chan(hyb, chan):
        res = load_tiff_stack(base_path + '{}/c{}.tif'.format(hyb, chan))
        res = res.astype(np.float32)
        res = res / 255
        return res

    hyb1 = [load_hyb_chan(1, c) for c in [2, 3, 4, 5]]
    hyb2 = [load_hyb_chan(2, c) for c in [2, 3, 4, 5]]
    hyb3 = [load_hyb_chan(3, c) for c in [2, 3, 4, 5]]
    hyb4 = [load_hyb_chan(4, c) for c in [2, 3, 4, 5]]

    stack = list_to_stack(hyb1 + hyb2 + hyb3 + hyb4)
    dapi = load_tiff_stack(base_path + 'DO/c1.tif')
    dots = load_tiff_stack(base_path + 'DO/c2.tif')

    return stack, dapi, dots


def filter_stack(stack, dots, disk_size):
    stack_filt = [white_top_hat(s, disk_size) for s in stack_to_list(stack)]
    stack_filt = list_to_stack(stack_filt)
    dots_filt = white_top_hat(dots, disk_size)

    return stack_filt, dots_filt


def register(stack, dots, upsample, comb_type='max'):
    hybs = dict()

    for j, k in enumerate([1, 5, 9, 13]):
        res = np.array([i for i in range(k, k + 4)])
        hybs[j] = res - 1

    max_projs = list_to_stack([max_proj(stack[inds, :, :]) for hyb, inds in hybs.iteritems()])
    mean_projs = list_to_stack([np.mean(stack[inds, :, :], axis=0) for hyb, inds in hybs.iteritems()])

    # set up (max_proj - mean_proj)*7
    comb = np.zeros(max_projs.shape)
    for k in range(4):
        res = (max_projs[k, :, :] - mean_projs[k, :, :]) * 7
        res[res <= 0] = 0
        comb[k, :, :] = res

    if comb_type == 'max':
        ref = max_projs
    else:
        ref = comb

    stack_reg = np.zeros(stack.shape)

    for k, inds in hybs.iteritems():
        shift, error = compute_shift(ref[k, :, :], dots, upsample)
        print "For hyb: {}, Shift: {}, Error: {}".format(k, shift, error)
        for i in inds:
            stack_reg[i, :, :] = shift_im(stack[i, :, :].astype(np.float32), shift)

    return stack_reg


def detect_spots(stack, dots, spot_sig=3, measurement_type='max'):
    stats = im_describe(dots)
    thresh = stats['mean'] + spot_sig * stats['std']

    s = BinarySpotDetector(stack, thresh, blobs=dots).detect(measurement_type)

    spots_df_tidy = s.to_encoder_dataframe(tidy_flag=True)
    spots_df_tidy = spots_df_tidy.ix[1:]

    spots_df_viz = s.to_viz_dataframe()

    spots_labels = s.labels

    return spots_df_tidy, spots_df_viz, spots_labels


def segment(dapi, dots):
    glp = dots / dots.max()

    dapi_thresh = .1
    stain_thresh = 0.06
    size_lim = (10, 500)
    disk_size_markers = None
    disk_size_mask = None

    seg = WatershedSegmenter(dapi / dapi.max(), glp)
    cells_labels = seg.segment(dapi_thresh, stain_thresh, size_lim, disk_size_markers, disk_size_mask)

    return cells_labels


def decode(spots_df_tidy):
    spots_df_tidy.loc[:, 'ch'] = None
    spots_df_tidy.loc[:, 'hyb'] = None

    for k in [1, 2, 3, 4]:
        cols = ['hyb_{}'.format(j) for j in [k, k + 4, k + 8, k + 12]]
        ind = spots_df_tidy.hybs.isin(cols)
        spots_df_tidy.loc[ind, 'ch'] = k

    for j, k in enumerate([1, 5, 9, 13]):
        cols = ['hyb_{}'.format(i) for i in range(k, k + 4)]
        ind = spots_df_tidy.hybs.isin(cols)
        spots_df_tidy.loc[ind, 'hyb'] = j + 1

    del spots_df_tidy['hybs']

    num_ch = 4
    num_hy = 4
    num_spots = spots_df_tidy.spot_id.max()

    seq_res = np.zeros((num_spots, num_hy))
    seq_stren = np.zeros((num_spots, num_hy))
    seq_qual = np.zeros((num_spots, num_hy))

    for sid in range(1, num_spots + 1):

        sid_df = spots_df_tidy[spots_df_tidy.spot_id == sid]

        mat = np.zeros((num_hy, num_ch))
        inds = zip(sid_df.hyb.values - 1, sid_df.ch.values - 1, sid_df.vals)

        for tup in inds:
            mat[tup[0], tup[1]] = tup[2]

        max_stren = np.max(mat, axis=1)
        max_ind = np.argmax(mat, axis=1)
        qual = max_stren / np.sum(mat, axis=1)

        seq_res[sid - 1, :] = max_ind
        seq_stren[sid - 1, :] = max_stren
        seq_qual[sid - 1, :] = qual

    max_qual = np.max(seq_qual, axis=1)

    letters = ['T', 'G', 'C', 'A']

    codes = []
    for k in range(seq_res.shape[0]):
        letter_inds = seq_res[k, :]
        letter_inds = letter_inds.astype(np.int)
        res = ''.join([letters[j] for j in letter_inds])
        codes.append(res)

    dec = pd.DataFrame({'spot_id': range(1, num_spots + 1),
                        'gene': codes,
                        'qual': max_qual})

    return dec


@click.command()
@click.option('--disk_size', default=15, help='White Top Hat Filter')
@click.option('--reg', default=0, prompt='Amount of registration')
@click.option('--reg_type', default='max', prompt='Max or Comb')
@click.option('--spot_sig', default=3.0, prompt='Spot threshold')
@click.option('--spot_type', default='max', prompt='Spot threshold')
def main(disk_size, reg, reg_type, spot_sig, spot_type):
    print 'loading data'
    stack, dapi, dots = load_data()

    print 'filtering'
    stack_filt, dots_filt = filter_stack(stack, dots, disk_size)

    print 'registering. upsample={}, type={}'.format(reg, reg_type)
    reg = int(reg)
    if reg == 0:
        stack_reg = stack_filt
    else:
        stack_reg = register(stack_filt, dots_filt, reg, comb_type=reg_type)

    print 'detecting spots. spot_sig: {}, spot_type: {}'.format(spot_sig, spot_type)

    spots_df_tidy, spots_df_viz, spots_labels = detect_spots(stack_reg, dots_filt, spot_sig, spot_type)

    print 'decoding'
    dec = decode(spots_df_tidy)

    top_gene = dec.gene.value_counts()[0:5]
    plt.figure()
    plt.hist(dec.qual, bins=20)
    plt.show()
    print top_gene


if __name__ == '__main__':
    main()
