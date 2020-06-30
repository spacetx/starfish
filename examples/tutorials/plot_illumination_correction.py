"""
.. _tutorial_illumination_correction:

Illumination Correction
=======================

The goal of illumination correction is to remove uneven illumination of the image caused by non
uniform illumination of the field of view, characteristics of the sensor, (like vignetting), or
orientation of the tissue’s surface with respect to the light source.

Prospective Correction
----------------------

The simplest forms of illumination correction are called “prospective correction” and are based on
background subtraction. This involves taking additional images using the microscopy apparatus to
help calibrate. These can either be acquired by averaging a series of images captured with no sample
and no light (dark image), or with no sample and light (bright image).

Starfish can apply this type of background correction by exposing the
:py:class:`.ElementWiseMultiply` :ref:`Filter <filtering>`. The user is responsible for transforming
their calibration images into the correct matrix to correct for background, and then
:py:class:`.ElementWiseMultiply` can apply a transformation to correct any uneven illumination.

The below plot shows how to use :py:class:`.ElementWiseMultiply` on a single plane of an in-situ
sequencing experiment.
"""

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
# be derived from the additional black or bright images taken to calibrate the microscope,
# and the correction is likely to be more more complex than a simple gradient.

ewm = starfish.image.Filter.ElementWiseMultiply(mult_array=gradient)
corrected_image_2d = ewm.run(image_2d, in_place=False)

###################################################################################################
# the image should now be returned to normal
plt.imshow(np.squeeze(corrected_image_2d.xarray))
plt.show()

###################################################################################################
# Retrospective Correction
# ------------------------
# When additional images were not acquired, or cannot be used calibrate the microscope,
# then uneven background illumination can be subtracted by estimating the background. This is
# called "retrospective correction". Low pass filters like :py:class:`.GaussianLowPass` and
# morphological filters are common ways to compute an approximate background image.
#
# A simple one-step process is to use the :py:class:`.WhiteTophat`, which will perform the
# background estimation and subtraction. See the :ref:`white_tophat` example in the Removing
# Autofluorescence tutorial.