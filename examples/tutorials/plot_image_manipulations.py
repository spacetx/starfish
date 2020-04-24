"""
Image Manipulations
===================
"""


###################################################################################################
# .. _tutorial_cropping:
#
# Cropping
# ========
# Starfish offers several options for cropping image data, both out-of-memory on load and in-memory.
# These can be useful for (1) restricting the size of a large image to load on demand and (2) removing
# edge effects after image processing approaches have been applied.
#
# Crop on Load
# ------------
# The first opportunity to subset data is during loading. Data is not loaded until
# :py:meth:`.get_image` is called on a :py:class:`.FieldOfView`. Here we demonstrate how to
# reduce the :code:`(y, x)` size of the :py:class:`.ImageStack` that is downloaded. Note that all
# of the data for the complete tiles must still be downloaded, but only the cropped dimensions
# will be loaded into memory.
#
# For this example we'll use a very small test image from an in-situ sequencing experiment.

import starfish
import starfish.data
from starfish.core.imagestack.parser.crop import CropParameters

experiment: starfish.Experiment = starfish.data.ISS(use_test_data=True)
field_of_view: starfish.FieldOfView = experiment["fov_001"]
print(field_of_view)

###################################################################################################
# printing the :py:class:`.FieldOfView` shows that the test dataset is :code:`(140, 200)`` pixels in
# :code:`(y, x)`. We'll load just the first :code:`(100, 80)` pixels to demonstrate starfish's
# crop-on-load functionality.

y_slice = slice(0, 100)
x_slice = slice(0, 80)
image: starfish.ImageStack = field_of_view.get_image("primary", x=x_slice, y=y_slice)

print(image)

###################################################################################################
# Note the reduced size of the image.
#
#
# Selecting Images
# ----------------
# Once an image has been loaded into memory as an :py:class:`.ImageStack` object, it is also
# possible to crop the :py:class:`.ImageStack` or select a subset of the images associated with the
# rounds, channels, and z-planes of the experiment.
#
# Here, we demonstrate selecting the last 50 pixels of (x, y) for rounds 2 and 3 using the
# :py:meth:`.ImageStack.sel` method.

from starfish.types import Axes

cropped_image: starfish.ImageStack = image.sel(
    {Axes.ROUND: (2, 3), Axes.X: (30, 80), Axes.Y: (50, 100)}
)
print(cropped_image)

###################################################################################################
# .. _tutorial_projection:
#
# Projection
# ==========
#
# In addition to selecting specific tiles, starfish can also project along an axis of the
# :py:class:`.ImageStack`, thereby reducing the size of the dimension to one. For example,
# projecting along the z-axis using the :code:`max` function will reduce a 3D image volume into a 2D
# image where the intensity value at each :code:`(y, x)` pixel is equal to the maximum of the
# voxel intensities at :code:`(y, x)` of the original image volume. This is a common approach for
# data that has relatively few spots. So long as the projection is unlikely to produce
# overlapping spots, projecting the data in this way can dramatically reduce processing time, as
# 2D algorithms are typically much faster than their 3D counterparts.
#
# Because the example image that we've downloaded has only one :code:`Axes.ZPLANE`,
# we will instead demonstrate the use of :py:meth:`.reduce` by projecting over
# :code:`Axes.CH` to produce an image of all the spots that appear in any channel for each
# round.

from starfish.image import Filter

projected_image: starfish.ImageStack = image.reduce({Axes.CH}, func="max")

###################################################################################################
# To demonstrate the effect, the below figure displays each channel of round :code:`1` in the
# left and center columns, and the maximum projection on the right.

import matplotlib.pyplot as plt
import xarray as xr

# select an image for plotting in 2d
round_1_ch_0: xr.DataArray = image.sel({Axes.CH: 0, Axes.ROUND: 1}).xarray.squeeze()
round_1_ch_1: xr.DataArray = image.sel({Axes.CH: 1, Axes.ROUND: 1}).xarray.squeeze()
round_1_ch_2: xr.DataArray = image.sel({Axes.CH: 2, Axes.ROUND: 1}).xarray.squeeze()
round_1_ch_3: xr.DataArray = image.sel({Axes.CH: 3, Axes.ROUND: 1}).xarray.squeeze()
round_1_proj: xr.DataArray = projected_image.sel({Axes.ROUND: 1}).xarray.squeeze()

# plot the images
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
ax1.imshow(round_1_ch_0)
ax1.set_title("round 1\nchannel 0")
ax2.imshow(round_1_ch_1)
ax2.set_title("round 1\nchannel 1")
ax4.imshow(round_1_ch_2)
ax4.set_title("round 1\nchannel 2")
ax5.imshow(round_1_ch_3)
ax5.set_title("round 1\nchannel 3")

ax3.imshow(round_1_proj)
ax3.set_title("round 1\nmaximum projection")

# we're not using the 6th plot
ax6.set_axis_off()

# fix matplotlib whitespace
f.tight_layout()
