import numpy as np

from starfish.stats import stack_stat


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
    stats = stack_stat(stack)
    ims = stack_to_list(stack)
    res = [im / s[metric] for im, s in zip(ims, stats)]
    return list_to_stack(res)
