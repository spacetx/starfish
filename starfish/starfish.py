import sys

import click
import matplotlib.pyplot as plt
import numpy as np
from showit import tile

from .filters import white_top_hat
from .io import Stack
from .register import compute_shift, shift_im

# mkdir /tmp/starfish/raw
# mkdir /tmp/starfish/formatted
# mkdir /tmp/starfish/registered
# mkdir /tmp/starfish/filtered
#
# python examples/get_iss_data.py /tmp/starfish/raw /tmp/starfish/formatted --d 1
#
# starfish register /tmp/starfish/formatted/org.json /tmp/starfish/registered/ --u 1000
#
# starfish filter /tmp/starfish/registered/org.json /tmp/starfish/filtered/ --ds 15
#
# starfish show /tmp/starfish/filtered/org.json
#
# rm -rf /tmp/starfish/raw
# rm -rf /tmp/starfish/formatted
# rm -rf /tmp/starfish/registered
# rm -rf /tmp/starfish/filtered

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
            # apply shift to all channels and hyb ronds
            res[h, c, :] = shift_im(s.data[h, c, :], shift)

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
@click.option('--sz', default=10, help='Figure size', type=int)
def show(in_json, sz):
    s = Stack()
    s.read(in_json)
    tile(s.squeeze(), size=sz, bar=True)
    plt.show()
