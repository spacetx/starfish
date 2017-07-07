from showit import tile

from starfish.stats import stack_describe


def tile_lims(stack, num_std, bar=True, size=20):
    stats = stack_describe(stack)
    lims = [s['mean'] + num_std * s['std'] for s in stats]
    tile(stack, bar=bar, clim=zip([0] * len(lims), lims), size=size)
