import skimage
from packaging import version

if version.parse(skimage.__version__) >= version.parse("0.16.0"):
    import skimage.exposure
    match_histograms = skimage.exposure.match_histograms
elif version.parse("0.16.0") > version.parse(skimage.__version__) > version.parse("0.14.2"):
    import skimage.transform
    match_histograms = skimage.transform.match_histograms
else:

    """
    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
    3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS`` AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    """

    import numpy as np

    def _match_cumulative_cdf(source, template):
        """
        Return modified source array so that the cumulative density function of
        its values matches the cumulative density function of the template.
        """
        src_values, src_unique_indices, src_counts = np.unique(
            source.ravel(), return_inverse=True, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_a_values[src_unique_indices].reshape(source.shape)

    def match_histograms(image, reference, multichannel=False):
        """
        Adjust an image so that its cumulative histogram matches that of another.
        The adjustment is applied separately for each channel.

        Parameters
        ----------
        image : ndarray
            Input image. Can be gray-scale or in color.
        reference : ndarray
            Image to match histogram of. Must have the same number of channels as
            image.
        multichannel : bool, optional
            Apply the matching separately for each channel.

        Returns
        -------
        matched : ndarray
            Transformed input image.

        Raises
        ------
        ValueError
            Thrown when the number of channels in the input image and the reference
            differ.

        References
        ----------
        .. [1] http://paulbourke.net/miscellaneous/equalisation/
        """
        if image.ndim != reference.ndim:
            raise ValueError('Image and reference must have the same number of channels.')

        if multichannel:
            if image.shape[-1] != reference.shape[-1]:
                raise ValueError('Number of channels in the input image and reference '
                                 'image must match!')

            matched = np.empty(image.shape, dtype=image.dtype)
            for channel in range(image.shape[-1]):
                matched_channel = _match_cumulative_cdf(
                    image[..., channel], reference[..., channel])
                matched[..., channel] = matched_channel
        else:
            matched = _match_cumulative_cdf(image, reference)

        return matched

if version.parse(skimage.__version__) > version.parse("0.14.2"):
    import skimage.feature
    blob_log = skimage.feature.blob_log
    blob_dog = skimage.feature.blob_dog
else:
    from skimage.feature.blob import _prune_blobs
    import numpy as np
    from scipy.ndimage import gaussian_filter, gaussian_laplace
    from skimage import img_as_float
    from skimage.feature import peak_local_max

    def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
                 overlap=.5, *, exclude_border=False):
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
            intensities.
        overlap : float, optional
            A value between 0 and 1. If the area of two blobs overlaps by a
            fraction greater than `threshold`, the smaller blob is eliminated.
        exclude_border : int or bool, optional
            If nonzero int, `exclude_border` excludes blobs from
            within `exclude_border`-pixels of the border of the image.

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

        # local_maxima = get_local_maxima(image_cube, threshold)
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

    def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
                 overlap=.5, log_scale=False, *, exclude_border=False):
        """
        Finds blobs in the given grayscale image.
        Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
        For each blob found, the method returns its coordinates and the standard
        deviation of the Gaussian kernel that detected the blob.

        Parameters
        ----------
        image : 2D or 3D ndarray
            Input grayscale image, blobs are assumed to be light on dark
            background (white on black).
        min_sigma : scalar or sequence of scalars, optional
            the minimum standard deviation for Gaussian kernel. Keep this low to
            detect smaller blobs. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number, in
            which case it is equal for all axes.
        max_sigma : scalar or sequence of scalars, optional
            The maximum standard deviation for Gaussian kernel. Keep this high to
            detect larger blobs. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number, in
            which case it is equal for all axes.
        num_sigma : int, optional
            The number of intermediate values of standard deviations to consider
            between `min_sigma` and `max_sigma`.
        threshold : float, optional.
            The absolute lower bound for scale space maxima. Local maxima smaller
            than thresh are ignored. Reduce this to detect blobs with less
            intensities.
        overlap : float, optional
            A value between 0 and 1. If the area of two blobs overlaps by a
            fraction greater than `threshold`, the smaller blob is eliminated.
        log_scale : bool, optional
            If set intermediate values of standard deviations are interpolated
            using a logarithmic scale to the base `10`. If not, linear
            interpolation is used.
        exclude_border : int or bool, optional
            If nonzero int, `exclude_border` excludes blobs from
            within `exclude_border`-pixels of the border of the image.

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
        .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

        Examples
        --------
        >>> from skimage import data, feature, exposure
        >>> img = data.coins()
        >>> img = exposure.equalize_hist(img)  # improves detection
        >>> feature.blob_log(img, threshold = .3)
        array([[ 266.        ,  115.        ,   11.88888889],
            [ 263.        ,  302.        ,   17.33333333],
            [ 263.        ,  244.        ,   17.33333333],
            [ 260.        ,  174.        ,   17.33333333],
            [ 198.        ,  155.        ,   11.88888889],
            [ 198.        ,  103.        ,   11.88888889],
            [ 197.        ,   44.        ,   11.88888889],
            [ 194.        ,  276.        ,   17.33333333],
            [ 194.        ,  213.        ,   17.33333333],
            [ 185.        ,  344.        ,   17.33333333],
            [ 128.        ,  154.        ,   11.88888889],
            [ 127.        ,  102.        ,   11.88888889],
            [ 126.        ,  208.        ,   11.88888889],
            [ 126.        ,   46.        ,   11.88888889],
            [ 124.        ,  336.        ,   11.88888889],
            [ 121.        ,  272.        ,   17.33333333],
            [ 113.        ,  323.        ,    1.        ]])

        Notes
        -----
        The radius of each blob is approximately :math:`sqrt{2}sigma` for
        a 2-D image and :math:`sqrt{3}sigma` for a 3-D image.
        """
        image = img_as_float(image)

        # if both min and max sigma are scalar, function returns only one sigma
        scalar_sigma = (
            True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
        )

        # Gaussian filter requires that sequence-type sigmas have same
        # dimensionality as image. This broadcasts scalar kernels
        if np.isscalar(max_sigma):
            max_sigma = np.full(image.ndim, max_sigma, dtype=float)
        if np.isscalar(min_sigma):
            min_sigma = np.full(image.ndim, min_sigma, dtype=float)

        # Convert sequence types to array
        min_sigma = np.asarray(min_sigma, dtype=float)
        max_sigma = np.asarray(max_sigma, dtype=float)

        if log_scale:
            start, stop = np.log10(min_sigma)[:, None], np.log10(max_sigma)[:, None]
            space = np.concatenate(
                [start, stop, np.full_like(start, num_sigma)], axis=1)
            sigma_list = np.stack([np.logspace(*s) for s in space], axis=1)
        else:
            scale = np.linspace(0, 1, num_sigma)[:, None]
            sigma_list = scale * (max_sigma - min_sigma) + min_sigma

        # computing gaussian laplace
        # average s**2 provides scale invariance
        gl_images = [-gaussian_laplace(image, s) * s ** 2
                     for s in np.mean(sigma_list, axis=1)]

        image_cube = np.stack(gl_images, axis=-1)

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
