from showit import image, tile


def tile_lims(stack, num_std, bar=True, size=20):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    lims = [s['mean'] + num_std * s['std'] for s in stats]
    tile(stack, bar=bar, clim=zip([0] * len(lims), lims), size=size)


def image_lims(im, num_std, bar=True, size=20):
    from starfish.stats import im_describe
    stats = im_describe(im)
    lim = stats['mean'] + num_std * stats['std']
    image(im, bar=bar, size=size, clim=[0, lim])
