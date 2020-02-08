"""
.. _howto_clip_value_to_zero:

Clipping Value To Zero
======================

How to use :py:class:`~starfish.image.Filter.ClipValueToZero` to clip high and low intensity
values of image planes or image volumes in an
:py:class:`~starfish.core.imagestack.imagestack.ImageStack`.

If you know the raw pixel values you want to use to clip images with instead of using a
percentile to determine the value you can use :py:class:`~starfish.image.Filter.ClipValueToZero`.
Any pixel values that fall outside the interval are clipped to the interval edges and the pixel
values are shifted such that the minimum value is set to zero.

This is best used when you have familiarity with your experiment data and have identified some
consistent pattern. For example, maybe you know the range of intensities and cutoffs that are
indicative of background and signal in each of the channels and you do not want to rely on the
percentile to pick the correct interval values.
"""

# Load the primary ImageStack from example DARTFISH data
import starfish.data
import matplotlib.pyplot as plt
from starfish.types import Axes, Levels
from starfish import FieldOfView
from starfish.image import Filter
from starfish.util.plot import intensity_histogram

experiment = starfish.data.DARTFISH(use_test_data=False)
stack = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

# Define some useful functions for viewing multiple images and histograms
def plot_intensity_histograms(stack: starfish.ImageStack, r: int, title: str):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title='ch: 0')
    ax2 = fig.add_subplot(132, title='ch: 1', sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title='ch: 2', sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title)

# View images and distribution of intensities in round 1
plot_intensity_histograms(stack, r=1, title='Distribution before clipping')

# Clip ImageStack without
clipper = Filter.ClipValueToZero(v_min= 0.00003, v_max= 0.001, level_method=Levels.SCALE_BY_CHUNK)
clipper.run(stack, in_place=True)

plot_intensity_histograms(stack, r=1, title='Distribution after clipping with value')