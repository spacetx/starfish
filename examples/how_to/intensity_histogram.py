"""
.. _tutorial_intensity_histogram:

Plotting Intensity Distribution
===============================

How to use :py:func:`~starfish.util.plot.intensity_histogram` to plot the intensity distribution of
any :py:class:`.ImageStack`.

The selector parameter can be used to pass a dictionary of dimensions and
:term:`indices<Index (Tile)>` to select a subset of the :py:class:`.ImageStack` for plotting.
Choosing the correct number of ``bins`` is important for accurately representing the distribution as
a histogram and it may be worth trying a couple different bin sizes.

The histogram is useful for examining image data and deciding how to normalize before
:term:`decoding<Decoding>`.

* :ref:`Normalizing Intensity Distributions <tutorial_normalizing_intensity_distributions>`
* :ref:`Normalizing Intensity Values <tutorial_normalizing_intensity_values>`
"""

# Load ImageStack from example DARTFISH data
import starfish.data
from starfish import FieldOfView
from starfish.types import Axes
experiment = starfish.data.DARTFISH(use_test_data=False)
stack = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(stack)

# Plot
import matplotlib
import matplotlib.pyplot as plt
from starfish.util.plot import intensity_histogram
matplotlib.rcParams["figure.dpi"] = 150
f, (ax1, ax2) = plt.subplots(ncols=2)
f.suptitle('Intensity Histogram')

# Plot intensity distribution of entire as a histogram with 50 bins
intensity_histogram(stack, sel={Axes.ROUND: 0, Axes.CH: 1}, log=True, bins=50, ax=ax1,
                    title='Full Image')

# Plot intensity distribution of 200x200 pixel ROI with 10 bins
intensity_histogram(stack, sel={Axes.ROUND: 0, Axes.CH: 1, Axes.X: (700, 900), Axes.Y: (100, 300),
                                Axes.ZPLANE: 0}, log=True, bins=10, ax=ax2, title='200px ROI')


