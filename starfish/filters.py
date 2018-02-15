import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from skimage import restoration
from skimage.filters import gaussian
from skimage.morphology import binary_erosion, binary_dilation, disk, binary_opening, binary_closing

from .munge import swap


def gaussian_low_pass(img, sigma):
    """
    Applies a gaussian low pass filter to an image. Implementation simply calls
    skimage.filters.gaussian
    :param img: Image to filter. :type numpy array
    :param sigma: Standard deviation of gaussian kernel: type int
    :return: Filtered image, same shape as input :type ndarray
    """
    img_swap = swap(img)

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


def gaussian_high_pass(img, sigma):
    """
    Applies a gaussian high pass filter to an image
    :param img: Image to filter. :type numpy array
    :param sigma: Standard deviation of gaussian kernel :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    blurred = gaussian_low_pass(img, sigma)

    over_flow_ind = img < blurred
    res = img - blurred
    res[over_flow_ind] = 0

    return res


def gaussian_kernel(shape=(3, 3), sigma=0.5):
    """
    Returns a gaussian kernel of specified shape and standard deviation.
    This is a simple python implementation of Matlab's fspecial('gaussian',[shape],[sigma])
    :param shape: Kernele shape. Default: (3,3) :type tuple of ints
    :param sigma: Standard deviation of gaussian kernel. Default: 0.5 :type int
    :return: Gaussian kernel. :type ndarray
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel


def richardson_lucy_deconv(img, num_iter, psf, clip=False):
    """
    Deconvolves input image with a specified point spread function. This simply calls
    skimage.restoration.richardson_lucy
    :param img: Image to filter. :type numpy array
    :param num_iter: Number of iterations to run algorithm :type int
    :param psf: Point spread function :type ndarray
    :param clip: False by default. If true, pixel value of the result above 1 or under -1 are thresholded
    for skimage pipeline compatibility. :type bool
    :return: Deconvolved image, same shape as input :type ndarray
    """

    img_deconv = restoration.richardson_lucy(img, psf, iterations=num_iter, clip=clip)

    # here be dragons. img_deconv is a float. this should not work, but the result looks nice
    # modulo boundary values? wtf indeed.
    img_deconv = img_deconv.astype(np.uint16)
    return img_deconv


def bin_erode(img, disk_size):
    """
    Performs binary erosion of an image
    :param img: Image to filter. :type numpy array
    :param disk_size: Radius of the disk-shaped structuring element.  :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    selem = disk(disk_size)
    res = binary_erosion(img, selem)
    return res


def bin_dilate(img, disk_size):
    """
    Performs binary dilation of an image
    :param img: Image to filter. :type numpy array
    :param disk_size: Radius of the disk-shaped structuring element.  :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    selem = disk(disk_size)
    res = binary_dilation(img, selem)
    return res


def bin_open(img, disk_size):
    """
    Performs binary opening of an image
    :param img: Image to filter. :type numpy array
    :param disk_size: Radius of the disk-shaped structuring element.  :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    selem = disk(disk_size)
    res = binary_opening(img, selem)
    return res


def bin_close(img, disk_size):
    """
    Performs binary closing of an image
    :param img: Image to filter. :type numpy array
    :param disk_size: Radius of the disk-shaped structuring element.  :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    selem = disk(disk_size)
    res = binary_closing(img, selem)
    return res


def bin_thresh(img, thresh):
    """
    Performs binary thresholding of an image
    :param img: Image to filter. :type numpy array
    :param thresh: Pixel values >= thresh are set to 1, else 0 :type int
    :return: Binarized image, same shape as input :type ndarray
    """
    res = img >= thresh
    return res


def white_top_hat(img, disk_size):
    """
    Performs white top hat filtering of an image to enhance spots
    :param img: Image to filter. :type numpy array
    :param disk_size: Radius of the disk-shaped structuring element.  :type int
    :return: Filtered image, same shape as input :type ndarray
    """
    selem = disk(disk_size)
    min_filt = minimum_filter(img, footprint=selem)
    max_filt = maximum_filter(min_filt, footprint=selem)
    res = img - max_filt
    return res
