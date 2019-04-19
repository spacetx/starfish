from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import skimage
import xarray as xr
from packaging import version
from scipy.ndimage import label, gaussian_filter, gaussian_laplace
from skimage import img_as_float
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs
from skimage.measure import regionprops
from sympy import Line, Point
from tqdm import tqdm

from starfish.config import StarfishConfig
from starfish.types import Axes, SpotAttributes
from starfish.types import Features


def _select_optimal_threshold(
        thresholds: np.ndarray, spot_counts: List[int], stringency: int
) -> float:

    # calculate the gradient of the number of spots
    grad = np.gradient(spot_counts)
    optimal_threshold_index = np.argmin(grad)

    # only consider thresholds > than optimal threshold
    thresholds = thresholds[optimal_threshold_index:]
    grad = grad[optimal_threshold_index:]

    # if all else fails, return 0.
    selected_thr = 0

    if len(thresholds) > 1:

        distances = []

        # create a line whose end points are the threshold and and corresponding gradient value
        # for spot_counts corresponding to the threshold
        start_point = Point(thresholds[0], grad[0])
        end_point = Point(thresholds[-1], grad[-1])
        line = Line(start_point, end_point)

        # calculate the distance between all points and the line
        for k in range(len(thresholds)):
            p = Point(thresholds[k], grad[k])
            dst = line.distance(p)
            distances.append(dst.evalf())

        # remove the end points
        thresholds = thresholds[1:-1]
        distances = distances[1:-1]

        # select the threshold that has the maximum distance from the line
        # if stringency is passed, select a threshold that is n steps higher, where n is the
        # value of stringency
        if distances:
            thr_idx = np.argmax(np.array(distances))

            if thr_idx + stringency < len(thresholds):
                selected_thr = thresholds[thr_idx + stringency]
            else:
                selected_thr = thresholds[thr_idx]

    return selected_thr


def _identify_local_maxima(
    image_cube, threshold, image, exclude_border, sigma_list, scalar_sigma, overlap
) -> np.ndarray:
    """Factor out local maxima calling

    Parameters
    ----------
    image_cube
    threshold
    image
    exclude_border
    sigma_list
    scalar_sigma
    overlap

    Returns
    -------

    """
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    return _prune_blobs(lm, overlap)


def blob_dog(
    image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
    num_thresholds=100, stringency=0, stop_at_spots=3, overlap=.5, *,
    exclude_border=False, verbose=False,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image.
    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities. If None, the function will attempt to calculate the
        optimal threshold by examining the gradient of the number of spots
        detected at each threshold, and selecting the stable threshold.
    num_thresholds : int, optional.
        If attempting to find the threshold, the number of linearly spaced
        thresholds to try between the minimum and maximum intensity of the
        image (inclusive).
    stringency : int, optional.
        If attempting to find the optimal threshold, step upward on the
        gradient of the number of spots detected a number of times equal to
        the stringency. Results in a higher threshold being selected.
    stop_at_spots : int, optional.
        If attempting to find the optimal threshold, stop attempting more
        stringent thresholds when this number of spots or fewer are detected.
        (default 3)
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    exclude_border : int or bool, optional
        If nonzero int, `exclude_border` excludes blobs from
        within `exclude_border`-pixels of the border of the image.
    verbose : bool
        If True, print progress

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

    Examples
    --------
    >>> from skimage import data, feature
    >>> feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    array([[ 267.      ,  359.      ,   16.777216],
        [ 267.      ,  115.      ,   10.48576 ],
        [ 263.      ,  302.      ,   16.777216],
        [ 263.      ,  245.      ,   16.777216],
        [ 261.      ,  173.      ,   16.777216],
        [ 260.      ,   46.      ,   16.777216],
        [ 198.      ,  155.      ,   10.48576 ],
        [ 196.      ,   43.      ,   10.48576 ],
        [ 195.      ,  102.      ,   16.777216],
        [ 194.      ,  277.      ,   16.777216],
        [ 193.      ,  213.      ,   16.777216],
        [ 185.      ,  347.      ,   16.777216],
        [ 128.      ,  154.      ,   10.48576 ],
        [ 127.      ,  102.      ,   10.48576 ],
        [ 125.      ,  208.      ,   10.48576 ],
        [ 125.      ,   45.      ,   16.777216],
        [ 124.      ,  337.      ,   10.48576 ],
        [ 120.      ,  272.      ,   16.777216],
        [  58.      ,  100.      ,   10.48576 ],
        [  54.      ,  276.      ,   10.48576 ],
        [  54.      ,   42.      ,   16.777216],
        [  52.      ,  216.      ,   16.777216],
        [  52.      ,  155.      ,   16.777216],
        [  45.      ,  336.      ,   16.777216]])

    Notes
    -----
    The radius of each blob is approximately :math:`sqrt{2}sigma` for
    a 2-D image and :math:`sqrt{3}sigma` for a 3-D image.
    """

    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * np.mean(sigma_list[i]) for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    if threshold is None:
        thresholds = np.linspace(image_cube.min(), image_cube.max(), num=num_thresholds)

        # number of spots detected at each threshold
        spot_counts = []

        # where we stop our threshold search
        stop_threshold = None

        if verbose and StarfishConfig().verbose:
            threshold_iter = tqdm(thresholds)
            print('Determining optimal threshold ...')
        else:
            threshold_iter = thresholds

        for stop_index, threshold in enumerate(threshold_iter):

            spots = _identify_local_maxima(
                image_cube, threshold, image, exclude_border, sigma_list, scalar_sigma, overlap
            )

            if len(spots) <= stop_at_spots:
                stop_threshold = threshold
                if verbose and StarfishConfig().verbose:
                    print(
                        f'Stopping early at threshold={threshold}. Number of spots fell below: '
                        f'{stop_at_spots}'
                    )
                break
            else:
                spot_counts.append(len(spots))

        if stop_threshold is None:
            stop_threshold = thresholds.max()

        if len(thresholds > 1):
            thresholds = thresholds[:stop_index]
            spot_counts = spot_counts[:stop_index]

        threshold = _select_optimal_threshold(thresholds, spot_counts, stringency)

    spots = _identify_local_maxima(
        image_cube, threshold, image, exclude_border, sigma_list, scalar_sigma, overlap
    )
    return spots
