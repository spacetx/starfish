"""
.. _tutorial_normalizing_intensity_values:

Normalizing Intensity Values
============================

It is important to normalize images before comparing intensity values between channels and rounds
to decode :term:`feature traces<Feature (Spot, Pixel) Trace>`. This tutorial will cover how to use
:py:class:`.ClipPercentileToZero` to normalize images within an :py:class:`.ImageStack`. For more
background on normalizing images in starfish pipelines see :ref:`section_normalizing_intensities`.

When the number of spots is not known to be uniform across :py:class:`.Axes` of an
:py:class:`.ImageStack`, you *cannot* use :ref:`MatchHistograms<tutorial_match_histograms>` to
normalize images. Instead, you can choose minimum and maximum percentile values or pixel intensity
values to clip outlier values and rescale the distributions to be more similar.

.. note::
    See the :ref:`Normalizing Intensity Distributions<tutorial_normalizing_intensity_distributions>`
    for when you know the number of spots is uniform.

This tutorial will focus on how to set parameters of :py:class:`.ClipPercentileToZero` to clip and
scale images from different channels but the same concepts can be applied to normalize images from
different rounds. Other uses for :py:class:`.ClipPercentileToZero` can be found in
:ref:`How To Clip Percentile To Zero<howto_clip_percentile_to_zero>`

The assumption in this example is that some of your images have fewer spots than others or no
spots at all so you can't use :py:class:`.MatchHistograms` to normalize intensities. But you still
need to normalize the intensities so that your :py:class:`.PerRoundMaxChannel` or
:py:class:`.MetricDistance` can accurately build a feature trace and decode. To do so you need to
know what your data looks like and what it should look like if every image had been acquired
under the exact same conditions.

The ideal intensity distributions after normalizing would be bimodal normal distribution mixtures
where the mean for each background distribution is the same and the mean for each spot signal
distribution is the same. The only difference would be the peak heights due to more or less
spots in an image.

.. note::
    :py:class:`.ClipValueToZero` and :py:class:`.Clip` can also be used in place of
    :py:class:`.ClipPercentileToZero`. See :ref:`howto_clip_value_to_zero` and :ref:`howto_clip`.

The first step to normalizing is viewing the histogram of pixel intensities.
"""

# Load the primary ImageStack from example DARTFISH data
import starfish.data
import matplotlib.pyplot as plt
from starfish.types import Axes, Levels
from starfish import FieldOfView
from starfish.image import Filter
from starfish.util.plot import imshow_plane, intensity_histogram

experiment = starfish.data.DARTFISH(use_test_data=False)
stack = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(stack)

# Define some useful functions for viewing multiple images and histograms
def imshow_3channels(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title='ch: 0')
    ax2 = fig.add_subplot(132, title='ch: 1')
    ax3 = fig.add_subplot(133, title='ch: 2')
    imshow_plane(stack, sel={Axes.ROUND: r, Axes.CH: 0}, ax=ax1)
    imshow_plane(stack, sel={Axes.ROUND: r, Axes.CH: 1}, ax=ax2)
    imshow_plane(stack, sel={Axes.ROUND: r, Axes.CH: 2}, ax=ax3)


def plot_intensity_histograms(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title='ch: 0')
    ax2 = fig.add_subplot(132, title='ch: 1', sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title='ch: 2', sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout()

# View images and distribution of intensities in round 1
imshow_3channels(stack=stack, r=1)
plot_intensity_histograms(stack=stack, r=1)

###################################################################################################
# The mean and max pixel intensity values are significantly lower in Channel 0 than the other two
# channels due to having fewer spots and having less PMT gain for that channel. The objective here
# is to normalize away differences like PMT gain but not alter the number of spots.
#
# The images here are 1024x1024 pixels so there are a total of ~1.05 million pixels. Channel 0 has
# almost 1 million pixels in the first bin, which is because the background is almost entirely
# 0 value pixels. Channel 1 has approximately 0.5 million pixels less in the first bin because
# it has low-intensity non-zero background.
#
# The first step in normalizing is removing the low-intensity background by clipping values below
# a minimum percentile ``p_min`` so that the each channel has approximately the same background
# distribution. The percentile to use depends on the percentage of the field of view that is
# background and what the range of pixel intensities is for background. A safe starting point for
# the images here is 80%.

# Clip values below 80%
clip_below_80 = Filter.Clip(p_min=80, p_max=100)
min80_clipped = clip_below_80.run(stack, in_place=False)
plot_intensity_histograms(stack=min80_clipped, r=1)

###################################################################################################
# Clipping the lowest 80% of pixels basically got the intended effect but because
# :py:class:`.Clip` was used the distribution does not start at zero.
#
# To fix that use :py:class:`.ClipPercentileToZero`. Also scale images with
# :py:class:`~starfish.types.Levels.SCALE_BY_CHUNK`, which scales every image plane between [0, 1].

# ClipPercentileToZero values below 80% and scale
cptz_1 = Filter.ClipPercentileToZero(p_min=80, p_max=100, level_method=Levels.SCALE_BY_CHUNK)
clipped_scaled = cptz_1.run(stack, in_place=False)
plot_intensity_histograms(stack=clipped_scaled, r=1)

###################################################################################################
# The histograms look pretty similar now and the difference in high intensity pixels is good
# because the channel 1 has a lot more spots than channel 0.
# The final step is is eliminating high-intensity outliers by clipping values above a maximum
# percentile ``p_max``, which allows the values of lower intensity spots to be scaled up to 1.
#
# You have to be very careful with ``p_max`` because you can unintentionally clip too much and scale
# background noise up to 1. Be conservative. To set ``p_max``, estimate the number of spots in the
# image with the fewest spots, and *be* *conservative*. If we say channel 0 has 10 spots, then
# 10 pixels out of 1.05 million pixels is the top 99.999 percentile, so ``p_max = 99.999``.

# ClipPercentileToZero values below 80% and above 99.999% and scale
cptz_2= Filter.ClipPercentileToZero(p_min=80, p_max=99.999, level_method=Levels.SCALE_BY_CHUNK)
clipped_both_scaled = cptz_2.run(stack, in_place=False)
plot_intensity_histograms(stack=clipped_both_scaled, r=1)

###################################################################################################
# These images have now been normalized.
#
# What if ``p_max`` was set to 99.99% instead?

# ClipPercentileToZero values below 80% and above 99.99% and scale
cptz_2= Filter.ClipPercentileToZero(p_min=80, p_max=99.99, level_method=Levels.SCALE_BY_CHUNK)
clipped_both_scaled = cptz_2.run(stack, in_place=False)
plot_intensity_histograms(stack=clipped_both_scaled, r=1)

####################################################################################################
# The histograms now match. If that was the goal for normalizing this data then great! But in this
# case it it an unintentional consequence of clipping too much.
#
# .. warning::
#     If your data is 3D with multiple z-planes, forgetting to set ``is_volume`` to True
#     will lead to incorrect clipping behavior, especially if ``level_method`` is set to
#     a :py:class:`.Levels` that rescales intensity values.
#


