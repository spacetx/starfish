import cv2
import numpy as np
from skimage import restoration
from skimage.morphology import binary_erosion, binary_dilation, disk, binary_opening, binary_closing


def gaussian_low_pass(img, sigma, ksize=None, border=None):
    img_swap = swap(img)
    if ksize is None:
        ksize = int(2 * np.ceil(2 * sigma) + 1)

    if border is None:
        border = cv2.BORDER_REPLICATE

    blurred = cv2.GaussianBlur(img_swap,
                               (ksize, ksize),
                               sigma,
                               borderType=border
                               )

    blurred = blurred.astype(np.uint16)
    return swap(blurred)


def gaussian_high_pass(img, sigma, ksize=None, border=None):
    blurred = gaussian_low_pass(img, sigma, ksize, border)

    over_flow_ind = img < blurred
    res = img - blurred
    res[over_flow_ind] = 0

    return res


def richardson_lucy_deconv(img, num_iter, psf=None, gpar=None, clip=False):
    if psf is None:
        if gpar is None:
            msg = 'Must specify a gaussian (kernel size, sigma) if a psf is not specified'
            raise ValueError(msg)
    else:
        ksize, sigma = gpar
        psf = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
        psf = np.dot(psf, psf.T)

    img_swap = swap(img)
    img_deconv = restoration.richardson_lucy(img_swap, psf, iterations=num_iter, clip=clip)

    # here be dragons. img_deconv is a float. this should not work, but the result looks nice
    # modulo boundary values? wtf indeed.
    img_deconv = img_deconv.astype(np.uint16)
    return swap(img_deconv)


def swap(img):
    img_swap = img.swapaxes(0, img.ndim - 1)
    return img_swap


def bin_eroode(im, disk_size):
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
