"""
.. _tutorial_match_histograms:

Matching Histograms
===================

How to use :py:class:`.MatchHistograms` to normalize intensity distributions across groups of
images in an :py:class:`.ImageStack`.

The ``group_by`` parameter can be set as one or more :py:class:`.Axes`. Images that share the
same :py:class:`.Axes` :term:`indices<Index (Tile)>` in ``group_by`` are grouped together. The
intensity distribution of each group is then quantile normalized to the mean intensity distribution.

Take for example an :py:class:`.ImageStack` with shape (r: 7, c: 4, z: 17, y: 1193, x: 913)

.. list-table:: Examples for ``group_by``
    :widths: 25 15 20 20
    :header-rows: 1

    * - group_by
      - number of groups
      - normalizes differences between
      - retains variability within
    * - Axes.CH
      - 4 groups
      - c
      - (r, x, y, z)
    * - Axes.CH, Axes.ROUND
      - 28 groups
      - c and r
      - (x, y, z)
    * - Axes.ZPLANE
      - 17 groups
      - z
      - (r, c, x, y)

"""

####################################################################################################
# How to match histograms across channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load ImageStack from example DARTFISH data
import starfish.data
from starfish import FieldOfView
from starfish.types import Axes
df_experiment = starfish.data.DARTFISH(use_test_data=False)
df_stack = df_experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(df_stack)

# Run MatchHistograms with group_by={Axes.CH}
mh_c = starfish.image.Filter.MatchHistograms({Axes.CH})
scaled_c = mh_c.run(df_stack, in_place=False, verbose=False, n_processes=8)


####################################################################################################
# How to match histograms across channels *and* rounds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Use loaded ImageStack from example DARTFISH data
# Run MatchHistograms with group_by={Axes.CH, Axes.ROUND}
mh_cr = starfish.image.Filter.MatchHistograms({Axes.CH, Axes.ROUND})
scaled_cr = mh_cr.run(df_stack, in_place=False, verbose=False, n_processes=8)

####################################################################################################
# How to match histograms across z-planes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load ImageStack from example BaristaSeq data
bs_experiment = starfish.data.BaristaSeq(use_test_data=False)
bs_stack = bs_experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(bs_stack)

mh_z = starfish.image.Filter.MatchHistograms({Axes.ZPLANE})
scaled_z = mh_z.run(bs_stack, in_place=False, verbose=False, n_processes=8)

####################################################################################################
# Risk of using MatchHistograms inappropriately
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To illustrate why matching histograms shouldn't be used to :ref:`normalize intensity
# distributions<tutorial_normalizing_intensity_distributions>` of images with significantly
# different number of spots you can see the result of matching histograms across z-planes from
# the previous example. The intensity histograms for ``Axes.ZPLANE: 2`` and ``Axes.ZPLANE: 8``
# are plotted before and after running :py:class:`.MatchHistograms`.
#
# ``Axes.ZPLANE: 2`` contains no spots or peaks so the histogram is that of Gaussian noise
#
# ``Axes.ZPLANE: 8`` has many spots and the histogram shows a long tail of high pixel values

# Plot intensity distributions of z-planes from z: 2 and z: 8 before and after scaling
import matplotlib
import matplotlib.pyplot as plt
from starfish.util.plot import intensity_histogram, imshow_plane
matplotlib.rcParams["figure.dpi"] = 150
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
f.suptitle('Intensity Histograms')

intensity_histogram(bs_stack, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 2}, log=True, bins=50, ax=ax1,
                    title='Unscaled\nz: 2')
intensity_histogram(bs_stack, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 8}, log=True, bins=50, ax=ax2,
                    title='Unscaled\nz: 8')
intensity_histogram(scaled_z, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 2}, log=True, bins=50, ax=ax3,
                    title='Scaled\nz: 2')
intensity_histogram(scaled_z, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 8}, log=True, bins=50, ax=ax4,
                    title='Scaled\nz: 8')
f.tight_layout()

####################################################################################################
# As expected, the distributions after scaling are made more similar. The higher values in the
# Gaussian noise are shifted higher while the long tail representing high spot intensities is
# reduced. Overall the SNR decreased. This does not mean it is never appropriate to
# :py:class:`.MatchHistograms` across :py:class:`.Axes.ZPLANE` but any use of
# :py:class:`.MatchHistograms` should be done so with caution.

