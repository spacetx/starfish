import numpy as np
import cv2


def gaussian_low_pass(img, sigma, ksize=None, border=None):
    if ksize is None:
        ksize = int(2 * np.ceil(2 * sigma) + 1)

    if border is None:
        border = cv2.BORDER_REPLICATE

    blurred = cv2.GaussianBlur(img,
                               (ksize, ksize),
                               sigma,
                               borderType=border
                               )
    return blurred.astype(np.int16)


def gaussian_high_pass(img, sigma, ksize=None, border=None):
    blurred = gaussian_low_pass(img, sigma, ksize, border)

    over_flow_ind = img < blurred
    res = img - blurred
    res[over_flow_ind] = 0

    return res
