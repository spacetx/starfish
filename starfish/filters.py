import cv2
import numpy as np
from skimage import restoration
from skimage.filters import gaussian
from skimage.morphology import binary_erosion, binary_dilation, disk, binary_opening, binary_closing
from scipy.ndimage.filters import maximum_filter, minimum_filter

from .munge import swap, stack_to_list, list_to_stack


def gaussian_low_pass(img, sigma, ksize=None, border=None, skimage=False):
    img_swap = swap(img)
    if ksize is None:
        ksize = int(2 * np.ceil(2 * sigma) + 1)

    if border is None:
        border = cv2.BORDER_REPLICATE

    if not skimage:
        blurred = cv2.GaussianBlur(img_swap,
                                   (ksize, ksize),
                                   sigma,
                                   borderType=border
                                   )
    else:

        blurred = gaussian(img_swap,
                           sigma=sigma,
                           output=None,
                           cval=0,
                           multichannel=True,
                           preserve_range=True,
                           truncate=4.0
                           )

    blurred = blurred.astype(np.uint16)

    return swap(blurred)


def gaussian_high_pass(img, sigma, ksize=None, border=None, skimage=False):
    blurred = gaussian_low_pass(img, sigma, ksize, border, skimage)

    over_flow_ind = img < blurred
    res = img - blurred
    res[over_flow_ind] = 0

    return res


def richardson_lucy_deconv(img, num_iter, psf=None, gpar=None, clip=False):
    if psf is None:
        if gpar is None:
            msg = 'Must specify a gaussian (kernel size, sigma) if a psf is not specified'
            raise ValueError(msg)
        ksize, sigma = gpar
        psf = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
        psf = np.dot(psf, psf.T)

    img_deconv = restoration.richardson_lucy(img, psf, iterations=num_iter, clip=clip)

    # here be dragons. img_deconv is a float. this should not work, but the result looks nice
    # modulo boundary values? wtf indeed.
    img_deconv = img_deconv.astype(np.uint16)
    return img_deconv


def bin_erode(im, disk_size):
    selem = disk(disk_size)
    res = binary_erosion(im, selem)
    return res


def bin_dilate(im, disk_size):
    selem = disk(disk_size)
    res = binary_dilation(im, selem)
    return res


def bin_open(im, disk_size):
    selem = disk(disk_size)
    res = binary_opening(im, selem)
    return res


def bin_close(im, disk_size):
    selem = disk(disk_size)
    res = binary_closing(im, selem)
    return res


def bin_thresh(im, thresh):
    res = im >= thresh
    return res


def white_top_hat(im, disk_size):
    selem = disk(disk_size)
    min_filt = minimum_filter(im, footprint=selem)
    max_filt = maximum_filter(min_filt, footprint=selem)
    res = im - max_filt
    return res
