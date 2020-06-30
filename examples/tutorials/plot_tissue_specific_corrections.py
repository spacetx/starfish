"""
.. _tutorial_removing_autoflourescence:

Removing Autofluorescence
=========================

In addition to the bright spots (signal) that we want to detect, microscopy experiments on tissue
slices often have a non-zero amount of auto-fluorescence from the cell bodies. This can be mitigated
by "clearing" strategies whereby tissue lipids and proteins are digested, or computationally by
estimating and subtracting the background values.

Here we demonstrate two computational ways to reduce background: clipping and white top-hat
filtering.
"""

###################################################################################################
# Clipping
# --------
# The simplest way to remove background is to set a global, (linear) cut-off and clip out the
# background values.

import starfish
import starfish.data
from starfish.image import Filter
from starfish.types import Axes

experiment: starfish.Experiment = starfish.data.ISS(use_test_data=True)
field_of_view: starfish.FieldOfView = experiment["fov_001"]
image: starfish.ImageStack = field_of_view.get_image("primary")

###################################################################################################
# Next, create the clip filter. Here we clip at the 97th percentile, optimally separates the spots
# from the background

clip_97 = Filter.Clip(p_min=97)
clipped: starfish.ImageStack = clip_97.run(image)

###################################################################################################
# plot both images

import matplotlib.pyplot as plt
import xarray as xr

# get the images
orig_plot: xr.DataArray = image.sel({Axes.CH: 0, Axes.ROUND: 0}).xarray.squeeze()
clip_plot: xr.DataArray = clipped.sel({Axes.CH: 0, Axes.ROUND: 0}).xarray.squeeze()

f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(orig_plot)
ax1.set_title("original")
ax2.imshow(clip_plot)
ax2.set_title("clipped")

###################################################################################################
# .. _white_tophat:
#
# White Top-Hat Filtering
# -----------------------
# Another way to remove background is to apply a white top-hat filter, which extracts small bright
# features, removes large foreground objects that we assume is unwanted noise (autofluorescence),
# and can correct for uneven background. The filter does this by estimating the background with a
# morphological opening on the image and then subtracting the background from the image.
#
# The white top-hat filter implemented in starfish uses a ball or disk structuring element for 3D
# or 2D images, respectively. The size of the structuring element can be thought of as the
# maximum size of foreground objects you want to keep. Anything larger will be removed. In the
# example below, a white top-hat filter is used to produce a similar effect to the clipping
# example without needing to select an intensity threshold.

from starfish.image import Filter

masking_radius = 5
filt = Filter.WhiteTophat(masking_radius, is_volume=False)
filtered = filt.run(image, verbose=True, in_place=False)

orig_plot: xr.DataArray = image.sel({Axes.CH: 0, Axes.ROUND: 0}).xarray.squeeze()
wth_plot: xr.DataArray = filtered.sel({Axes.CH: 0, Axes.ROUND: 0}).xarray.squeeze()

f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(orig_plot)
ax1.set_title("original")
ax2.imshow(wth_plot)
ax2.set_title("wth filtered")