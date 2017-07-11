from showit import tile

def tile_lims(stack, num_std, bar=True, size=20):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    lims = [s['mean'] + num_std * s['std'] for s in stats]
    tile(stack, bar=bar, clim=zip([0] * len(lims), lims), size=size)