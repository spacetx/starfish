"""
.. _howto_clip:

Clipping
========

How to use :py:class:`starfish.image.Filter.Clip` to clip high and low intensity values of image
planes or image volumes in an :py:class:`ImageStack` and rescale intensity values.

:py:class:`starfish.image.Filter.Clip` is useful for normalizing images, removing background, and removing
high-intensity outliers. If you want the values to start from zero after clipping see
:py:class:`ClipPercentileToZero`. Both :py:class:`AlgorithmFilter'\s use percentiles to set the
min and max values to clip.
"""

# Load :py:class:`ImageStack` from example BaristaSeq data
import starfish.data
import matplotlib.pyplot as plt
from starfish.types import Axes, Levels
from starfish import FieldOfView
from starfish.image import Filter
from starfish.util.plot import imshow_plane, intensity_histogram

bs_experiment = starfish.data.BaristaSeq(use_test_data=False)
stack = bs_experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

# Define some useful functions for viewing multiple images and histograms
def plot_intensity_histograms(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title='ch: 0')
    ax2 = fig.add_subplot(132, title='ch: 1', sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title='ch: 2', sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout()

# View distribution of intensities in round 0
plot_intensity_histograms(stack=stack, r=0)

# Clip imagestack with scaling
clipper = Filter.Clip(p_min=50, p_max=99.9, is_volume=True, level_method=Levels.SCALE_BY_CHUNK)
clipper.run(stack, in_place=True)

# View distribution of intensities after clipping
plot_intensity_histograms(stack, r=0)