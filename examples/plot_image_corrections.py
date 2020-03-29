"""
Image Corrections
=================
.. container:: toggle

    .. container:: header

        **Show/Hide Code**

    .. code-block:: xml
       :linenos:

       import matplotlib.pyplot as plt
       import numpy as np
       import xarray as xr

       import starfish
       import starfish.data
       from starfish.types import Axes

       experiment = starfish.data.ISS(use_test_data=True)
       image: starfish.ImageStack = experiment['fov_001'].get_image('primary')

       image_2d = image.sel({Axes.CH: 0, Axes.ROUND: 0, Axes.ZPLANE: 0})

       plt.imshow(np.squeeze(image_2d.xarray.values))
       plt.show()
"""


###################################################################################################
# .. _tutorial_illumination_correction:
#
# Illumination Correction
# =======================
#
# The goal of illumination correction is to remove uneven illumination of the image caused by non
# uniform illumination of the field of view, characteristics of the sensor, (like vignetting), or
# orientation of the tissue’s surface with respect to the light source.
#
# The simplest forms of illumination correction are called “prospective correction” and are based on
# background subtraction. This involves taking additional images using the microscopy apparatus to
# help calibrate. These can either be acquired by averaging a series of images captured with no sample
# and no light (dark image), or with no sample and light (bright image).
#
# Starfish can apply this type of background correction by exposing the :py:class:`ElementWiseMult`
# :ref:`Filter <filtering>`. The user is responsible for transforming their calibration images into
# the correct matrix to correct for background, and then :py:class:`ElementWiseMult` can apply a
# transformation to correct any uneven illumination.
#
# The below plot shows a single plane of an in-situ sequencing experiment.
#
# .. container:: toggle
#
#    .. container:: header
#
#       **Show/Hide Code**

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import starfish
import starfish.data
from starfish.types import Axes

experiment = starfish.data.ISS(use_test_data=True)
image: starfish.ImageStack = experiment['fov_001'].get_image('primary')

image_2d = image.sel({Axes.CH: 0, Axes.ROUND: 0, Axes.ZPLANE: 0})

plt.imshow(np.squeeze(image_2d.xarray.values))
plt.show()

###################################################################################################
# This image was corrected before it was sent to us, but we can introduce an uneven illumination
# profile. Below we mock an extremely severe 200% decrease in illumination from left to right.

lightness = np.linspace(4, 1, image_2d.xarray.sizes[Axes.X])
gradient_data = np.tile(lightness, reps=(image_2d.xarray.sizes[Axes.Y], 1))
gradient = xr.DataArray(
    data=gradient_data[np.newaxis, np.newaxis, np.newaxis, :, :],
    dims=(Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value, Axes.Y.value, Axes.X.value)
)

# introduce the gradient, overwriting the ImageStack
data = image_2d.xarray.values / gradient.values
image_2d = starfish.ImageStack.from_numpy(data)

# display the resulting image
plt.imshow(np.squeeze(image_2d.xarray.values))
plt.show()

###################################################################################################
# The illumination profile has increased the intensity of the background in the right side of the
# image. This is problematic for many spot finding methods that set thresholds for peak intensities
# globally across the image; spots can be incorrectly excluded in low-illumination areas, and this
# spatial phenomenon can lead to incorrect spatial hypotheses.
#
# We use starfish's ElementWiseMultiply to multiply the image with a gradient. Here, it's just the
# same gradient we divided the image by. However, in typical microscopy experiments this should
# be derived from black or bright images taken to calibrate the microscope, and the correction
# is likely to be more more complex than a simple gradient.

ewm = starfish.image.Filter.ElementWiseMultiply(mult_array=gradient)
corrected_image_2d = ewm.run(image_2d, in_place=False)

###################################################################################################
# the image should now be returned to normal
plt.imshow(np.squeeze(corrected_image_2d.xarray))
plt.show()

###################################################################################################
# .. _tutorial_chromatic_aberration:
#
# Chromatic Aberration
# ====================
#
# Chromatic Aberration refers to the failure of a lens to focus all colors to the same positions.
# Because multiplex spot counting experiments tend to leverage fluorescence signals from different
# spectral bands and restrict each color channel to its own image, the resulting problems can be
# complex to detect.
#
# Starfish currently exposes some basic registration effects, and these can be enough to correct for
# very minor chromatic aberrations. Additionally, non-multiplex approaches may only require that spots
# find their way into the correct cell, providing some flexibility to ignore minor aberrations.
#
# However, most multiplex experiments will require some kind of correction beyond what is provided by
# starfish's basic translation. At this point in time starfish **does not** provide tooling for the
# correction of chromatic aberrations, and as such users must correct for these types of errors
# in their data prior to submitting it to starfish. However, we would be very excited to receive
# code contributions of filters that solve these problems
# (see :ref:`contributing to starfish <contributing>`)
#
# To read more about types of chromatic aberration that can appear in microscopy data, see
# `wikipedia`_
#
# .. _wikipedia: https://en.wikipedia.org/wiki/Chromatic_aberration
#

pass

###################################################################################################
# .. _tutorial_deconvolution:
#
# Deconvolution of Optical Point Spread Functions
# ===============================================
#
# Deconvolution is a technique that enables a user to reverse any optical distortion introduced by
# the microscope. Deconvolution is accomplished by assuming that the path of light through the
# instrument is perfect, but convolved with a "point spread function". By deconvolving the image and
# the point spread function, the distortion can be removed.
#
# The point spread function can be determined in several ways. Ideally, it is approximated during
# calibration of the microscope, in which case it can be removed by the
# `Richardson-Lucy algorithm <richardson-lucy-web>`_ (API:
# :py:class:`~starfish.image.Filter.DeconvolvePSF`)
#
# .. _richardson-lucy-web: https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
#
# Learning the PSF from image data
# --------------------------------
#
# The simplest way to learn the PSF from spot-based data is to learn the shape of the spots. If we
# assume that the spots are derived from point sources of light, the average spot shape can form the
# basis for estimating the PSF needed by the richardson-lucy (see above). This depends on the data
# having a relatively uniform spot shape, which not all image-based transcriptomics and relatively few
# image-based proteomics experiments adhere to.
#
# # TODO incorporate `Nick's vignette <nsofroniew-vignette>`_.
#
# .. _nsofroniew-vignette: https://gist.github.com/sofroniewn/8acccc0d040fc8d9325267c83c8febc9
#
# It can also be estimated from the frequency properties of the experimental data, called
# "blind deconvolution", for which variants of both `Richardson-Lucy <richardson-lucy-blind>`_
# and the `Weiner filter <wiener>`_ have been proposed. Starfish does not provide any tooling
# for blind deconvolution.
#
# .. _richardson-lucy-blind: 10.1364/JOSAA.12.000058
# .. _wiener: https://www.sciencedirect.com/science/article/pii/S0377042717306544
#
# To read more about image deconvolution, see this article:
# `<https://en.wikipedia.org/wiki/Deconvolution#Optics_and_other_imaging>`
#
# Starfish

pass

###################################################################################################
# .. _tutorial_image_registration:
#
# Image Registration
# ==================
#
# Registration is an important aspect of image correction, particularly for multiplex experiments
# which attempt to match dots across images and channels. In these experiments, even very small shifts
# the size of a dot can make it extremely challenging to identify gene that dot present across a set
# of images.
#
# As mentioned above, chromatic aberration is one form of error that spot-calling experiments must
# contend with. However, registration error can also occur from tissue handling during the experiment,
# subtle changes in the position of the microscope slide relative to the stage, microfluidics shifting
# the tissue during fluid exchange across rounds, or changes in tissue morphology in expansion
# microscopy experiments.
#
# Starfish exposes simple fourier-domain translational registration to adjust for *some* of the above
# registration issues. Starfish also supports the Warp functionality to apply any
# pre-learned affine transformation, The combination of these approaches covers all of the types of
# registration faults that starfish's developers have observed thus far while looking at image-based
# transcriptomics experiments. However, it will be necessary to learn registrations that are more
# complex than translational shifts using software outside starfish, for the time being. Any
# contributions that add registration features would be highly desirable.
#
# TODO ambrosejcarr examples and links to warp
#

pass

###################################################################################################
# .. _tutorial_image_correction_pipeline:
#
# Example Image Correction Pipeline
# =================================
#
# TODO put together a worked example in the gallery and link it here.
