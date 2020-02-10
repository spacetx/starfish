"""
.. _tutorial_normalizing_intensity_distributions:

Normalizing Intensity Distributions
===================================

It is important to normalize images before comparing intensity values between channels and
rounds to decode :term:`feature traces<Feature (Spot, Pixel) Trace>`. This tutorial will teach you
when and how to normalize the *distributions* of intensities from images in an
:py:class:`.ImageStack`. For more background on normalizing images in starfish pipelines see
:ref:`section_normalizing_intensities`.

Normalizing the distributions is done in starfish by matching the histograms of
:py:class:`.ImageStack`\s to a reference histogram. The reference histogram is created by
averaging the histograms from each group defined by the ``group_by`` parameter. These groups also
determine along which :py:class:`.Axes` the intensities will be normalized.

Before normalizing, you need to know whether it is appropriate to make the intensity
distributions the same. For example, it would be incorrect to match histograms of an image with
*few* RNA spots and an image with *many* RNA spots because the histograms *should* look different.
However, if you know every channel of every round has a uniform abundance of spots and you want to
normalize for differences like fluorophore brightness and temporal changes of the microscope,
then matching histograms is a good choice for normalizing images and does not require setting
parameters.

.. note::
    See the :ref:`Normalizing Intensity Values<tutorial_normalizing_intensity_values>` tutorial for
    when you can't assume distributions should match after normalization.

This tutorial will demonstrate how to normalize intensity distributions on a data set that would
not typically be appropriate to normalize with :py:class:`.MatchHistograms` due to its particular
codebook design. The reason for using this data set here is to emphasize the effect of normalizing.
"""

# Load the primary images ImageStack from example DARTFISH data
import starfish.data
import matplotlib.pyplot as plt
from starfish.types import Axes
from starfish import FieldOfView
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


def plot_intensity_histograms(
        ref: starfish.ImageStack, scaled_cr: starfish.ImageStack, scaled_c: starfish.ImageStack,
        scaled_r: starfish.ImageStack, r: int):
    fig = plt.figure()
    ax10 = fig.add_subplot(4, 3, 10)
    intensity_histogram(scaled_cr, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax10)
    ax10.set_ylabel('ch and r', rotation=90, size='large')
    ax11 = fig.add_subplot(4, 3, 11, sharex=ax10)
    intensity_histogram(scaled_cr, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax11)
    ax12 = fig.add_subplot(4, 3, 12, sharex=ax10)
    intensity_histogram(scaled_cr, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax12)
    ax1 = fig.add_subplot(4, 3, 1, sharex=ax10)
    intensity_histogram(ref, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    ax1.set_title('ch: 0')
    ax1.set_ylabel('unscaled', rotation=90, size='large')
    ax2 = fig.add_subplot(4, 3, 2, sharex=ax10)
    intensity_histogram(ref, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    ax2.set_title('ch: 1')
    ax3 = fig.add_subplot(4, 3, 3, sharex=ax10)
    intensity_histogram(ref, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    ax3.set_title('ch: 2')
    ax4 = fig.add_subplot(4, 3, 4, sharex=ax10)
    intensity_histogram(scaled_c, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax4)
    ax4.set_ylabel('ch', rotation=90, size='large')
    ax5 = fig.add_subplot(4, 3, 5, sharex=ax10)
    intensity_histogram(scaled_c, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax5)
    ax6 = fig.add_subplot(4, 3, 6, sharex=ax10)
    intensity_histogram(scaled_c, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax6)
    ax7 = fig.add_subplot(4, 3, 7, sharex=ax10)
    intensity_histogram(scaled_r, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax7)
    ax7.set_ylabel('r', rotation=90, size='large')
    ax8 = fig.add_subplot(4, 3, 8, sharex=ax10)
    intensity_histogram(scaled_r, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax8)
    ax9 = fig.add_subplot(4, 3, 9, sharex=ax10)
    intensity_histogram(scaled_r, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax9)
    fig.tight_layout()

####################################################################################################
# By looking at unscaled images we see channel 0 has considerably less signal than channel 1 and
# typically we would not want to normalize their intensity distributions with
# :py:class:`.MatchHistograms`.
#
# .. note::
#     If your :py:class:`.ImageStack` has multiple z-planes you must select a z-plane or use
#     :py:meth:`.ImageStack.reduce` to use display the image
imshow_3channels(stack=stack, r=0)

####################################################################################################
# We create and run three :py:class:`.MatchHistograms` using different
# :py:class:`.Axes` groups to see the differences in normalizing.
#
# * ``group_by={Axes.CH, Axes.ROUND}`` will make intensity histograms of every (x,y,z) volume match
# * ``group_by={Axes.CH}`` will make intensity histograms of every (x,y,z,r) volume match
# * ``group_by={Axes.ROUND}`` will make intensity histograms of every (x,y,z,c) volume match
#

# MatchHistograms group_by channel and round
mh_cr = starfish.image.Filter.MatchHistograms({Axes.CH, Axes.ROUND})
# MatchHistograms group_by channel
mh_c = starfish.image.Filter.MatchHistograms({Axes.CH})
# MatchHistograms group_by round
mh_r = starfish.image.Filter.MatchHistograms({Axes.ROUND})
# Run MatchHistograms
scaled_cr = mh_cr.run(stack, in_place=False, verbose=False, n_processes=8)
scaled_c = mh_c.run(stack, in_place=False, verbose=False, n_processes=8)
scaled_r = mh_r.run(stack, in_place=False, verbose=False, n_processes=8)

####################################################################################################
# Plotting the intensity histograms shows the effect of normalization.
#
# * unscaled histograms of the three channels reflect the raw images -- channel 0 has less signal
# * normalizing with ``group_by={Axes.CH}`` has the effect of significantly rescaling histograms of channel 0 to match histograms of the other channels
# * normalizing with ``group_by={Axes.R}`` does not scale histograms of channel 0 to match histograms of other channels
# * normalizing with ``group_by={Axes.CH, Axes.ROUND}`` scales histograms from every round and channel to match each other
plot_intensity_histograms(ref=stack, scaled_cr=scaled_cr, scaled_c=scaled_c, scaled_r=scaled_r, r=0)