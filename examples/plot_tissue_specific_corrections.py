"""
Tissue Corrections
==================
"""

###################################################################################################
# .. _tutorial_removing_autoflourescence:
#
# Removing autofluorescence
# =========================
#
# In addition to the bright spots (signal) that we want to detect, microscopy experiments on tissue
# slices often have a non-zero amount of auto-fluorescence from the cell bodies. This can be mitigated
# by "clearing" strategies whereby tissue lipids and proteins are digested, or computationally by
# estimating and subtracting the background values.
#
# We use the same test image from the previous section to demonstrate how this can work.
#
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
# Next, create the clip filter. Here we clip at the 50th percentile, optimally separates the spots
# from the background

clip_50 = Filter.Clip(p_min=97)
clipped: starfish.ImageStack = clip_50.run(image)

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
#
