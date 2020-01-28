"""
.. _tutorial_normalizing_intensity_values:

Normalizing Intensity Values
============================

It is important to normalize images before comparing intensity values between
channels and rounds to create a feature trace. Without normalizing, the feature trace could be
biased by high pixel intensity values due to chemistry or microscopy factors unrelated to RNA
abundance. This tutorial will cover how to use py:class:`ClipPercentileToZero`,
:py:class:`ClipValueToZero`, and :py:class:`Clip` to normalize images within an
:py:class:`ImageStack`.

When the number of spots is not known to be uniform across :py:class:`Axes` of an
:py:class:`ImageStack`, you *cannot* use :ref:`MatchHistograms<tutorial_match_histograms>` to
normalize images. Instead, you can choose minimum and maximum percentile values or pixel intensity
values to remove outlier values and rescale the distributions to be more similar.

.. note::
    See the :ref:`Normalizing Intensity Distributions<tutorial_normalizing_intensity_distributions>`
    for when you know the number of spots is uniform.

This tutorial is divided into three sections, one for each clipping :py:class:`FilterAlgorithm`.
To build an image processing pipeline picking one section to follow should be sufficient for
normalizing images. However there is more than one way to process images and you may find a
different combination of :py:class:`FilterAlgorithm`s works best for your data.
This tutorial will use :ref:`intensity_histogram<tutorial_intensity_histogram>` and
:ref:`imshow_plane<tutorial_imshow_plane>` to visualize the image intensities before and after
clipping.
"""

####################################################################################################
# Clip Percentile To Zero
# =======================
#
# The recommended :py:class:`FilterAlgorithm` for clipping is :py:class:`ClipPercentileToZero`
# for most users. An interval is defined by pixel values at the p_min and p_max percentiles of
# the distribution. The interval bounds can be also scaled by min_coeff and max_coeff. Any pixel
# values that fall outside the interval are clipped to the interval edges. Lastly, the pixel
# values are shifted such that the minimum value is set to zero.
#
# Setting p_min is useful for

# Load the primary images :py:class:`ImageStack` from example BaristaSeq data


####################################################################################################
# Clip Value To Zero
# ==================
#
# If you know the raw pixel values you want to clip image values with instead of using a
# percentile to find the value you can use :py:class:`ClipValueToZero`. Any pixel values that
# fall outside the interval are clipped to the interval edges and the pixel values are shifted
# such that the minimum value is set to zero.
#
# This is best used when you have familiarity with your experiment data and have identified some
# consistent pattern. For example, maybe you know the range of intensities and cutoffs that are
# indicative of background and signal in each of the channels and you do not want to rely on the
# percentile to pick the correct interval values.
#
# How to set parameters
# v_min: a conservative estimate of pixel values you are background - these pixels will become zero
# v_max:



####################################################################################################
# Clip
# ====
#
# If you know the raw pixel values you want to clip image values with instead of using a
# percentile to find the value you can use :py:class:`ClipValueToZero`. Any pixel values that
# fall outside the interval are clipped to the interval edges and the pixel values are shifted
# such that the minimum value is set to zero.
#
# This is best used when you have familiarity with your experiment data and have identified some
# consistent pattern. For example, maybe you know the range of intensities and cutoffs that are
# indicative of background and signal in each of the channels and you do not want to rely on the
# percentile to pick the correct interval values.
#
# How to set parameters
# v_min: a conservative estimate of pixel values you are background - these pixels will become zero
# v_max: