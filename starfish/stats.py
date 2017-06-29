import numpy as np
import pandas as pd


def im_stat(im):
    shape = im.shape
    flat_dims = reduce(lambda x, y: x * y, shape)
    flat_im = np.reshape(im, flat_dims)
    stats = pd.Series(flat_im).describe()
    return stats.to_dict()


def stack_stat(stack):
    num_hybs = stack.shape[0]
    stats = [im_stat(stack[k, :]) for k in range(num_hybs)]
    return stats
