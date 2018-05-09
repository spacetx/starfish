import numpy as np

from .stats import im_describe, stack_describe


def tile_lims(stack, num_std, bar=True, size=20):
    from showit import tile
    if type(stack) is list:
        stack = np.array(stack)
    stats = stack_describe(stack)
    lims = [s['mean'] + num_std * s['std'] for s in stats]
    tile(stack, bar=bar, clim=list(zip([0] * len(lims), lims)), size=size)


def image_lims(im, num_std, bar=True, size=20):
    from showit import image
    stats = im_describe(im)
    lim = stats['mean'] + num_std * stats['std']
    image(im, bar=bar, size=size, clim=[0, lim])
