import numpy as np
import pandas as pd

from starfish.stats import stack_describe


def swap(img):
    img_swap = img.swapaxes(0, img.ndim - 1)
    return img_swap


def stack_to_list(stack):
    num_ims = stack.shape[0]
    return [stack[im, :] for im in range(num_ims)]


def list_to_stack(list):
    return np.array(list)


def max_proj(stack):
    im = np.max(stack, axis=0)
    return im


def scale(stack, metric, clip=False):
    stats = stack_describe(stack)
    ims = stack_to_list(stack)
    res = [im / s[metric] for im, s in zip(ims, stats)]
    return list_to_stack(res)


def gather(df, key, value, cols):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )
