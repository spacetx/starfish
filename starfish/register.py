import numpy as np
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation

from .munge import stack_to_list, list_to_stack


def compute_shift(im, ref, upsample_factor=1):
    shift, error, diffphase = register_translation(im, ref, upsample_factor)
    return shift, error


def shift_im(im, shift):
    fim_shift = fourier_shift(np.fft.fftn(im), map(lambda x: -x, shift))
    im_shift = np.fft.ifftn(fim_shift)
    return im_shift.real


def register(im, ref, upsample_factor=1):
    shift, _ = compute_shift(im, ref, upsample_factor)
    res = shift_im(im, shift)
    return res


def register_stack(stack, ref, upsample_factor=1):
    ims = stack_to_list(stack)
    res = [register(i, ref, upsample_factor) for i in ims]
    return list_to_stack(res)
