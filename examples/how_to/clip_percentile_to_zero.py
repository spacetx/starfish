"""
.. _howto_clip_percentile_to_zero:

Clipping Percentile To Zero
===========================

How to use :py:class:`.ClipPercentileToZero` to clip high and low intensity values of image
planes or image volumes in an :py:class:`.ImageStack`.

:py:class:`.ClipPercentileToZero` is the recommended :py:class:`.FilterAlgorithm` for clipping
images for most users. An interval is defined by pixel values at the ``p_min`` and ``p_max``
percentiles of the distribution. The interval bounds can also be scaled by multiplying with a
``min_coeff`` and ``max_coeff``. Any pixel values that fall outside the interval are clipped to
the interval edges. Lastly, the pixel values are shifted such that the minimum value is set to
zero. See :py:class:`.Clip` if you don't want values to be shifted to zero.

The minimum percentile ``p_min`` is useful for removing low-intensity background by setting
everything below ``p_min`` to zero. The maximum percentile ``p_max`` is useful for eliminating
high-intensity outliers. If your :py:class:`.ImageStack` has
greater than one z-plane, it is critical to set ``is_volume=True`` to get the expected clipping
behavior.

To see how :py:class:`.ClipPercentileToZero` can be used for normalizing images see
:ref:`tutorial_normalizing_intensity_values`.
"""


# Load ImageStack from example BaristaSeq data
import starfish.data
import matplotlib.pyplot as plt
from starfish.types import Axes, Levels
from starfish import FieldOfView
from starfish.image import Filter
from starfish.util.plot import imshow_plane, intensity_histogram

bs_experiment = starfish.data.BaristaSeq(use_test_data=False)
stack = bs_experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

# Define some useful functions for viewing multiple images and histograms
def plot_intensity_histograms(stack: starfish.ImageStack, r: int, title: str):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title='ch: 0')
    ax2 = fig.add_subplot(132, title='ch: 1', sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title='ch: 2', sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout()
    fig.suptitle(title)

# View distribution of intensities in round 1
plot_intensity_histograms(stack=stack, r=1, title='Distribution before clipping')

# Clip imagestack without setting is_volume to True
bad_clipper = Filter.ClipPercentileToZero(p_min=90, p_max=99.99, level_method=Levels.SCALE_BY_CHUNK)
bad_stack = bad_clipper.run(stack)

# View distribution if you forget to set is_volume
plot_intensity_histograms(bad_stack, r=1, title='Distribution after clipping with is_volume=False')

# Clip imagestack without setting is_volume to True
clipper = Filter.ClipPercentileToZero(p_min=90, p_max=99.99, is_volume=True, level_method=Levels.SCALE_BY_CHUNK)
clipper.run(stack, in_place=True)

# View distribution
plot_intensity_histograms(stack, r=1, title='Distribution after clipping with is_volume=True')