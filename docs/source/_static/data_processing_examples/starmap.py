"""
STARmap processing example
==========================

This notebook demonstrates the processing of STARmap data using starfish. The
data we present here is a subset of the data used in this
`publication <starmap>`_ and was generously provided to us by the authors.

.. _starmap: https://doi.org/10.1126/science.aat5691

Load example data
-----------------
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import starfish
import starfish.data
from starfish.types import Axes
from starfish.util.plot import imshow_plane

experiment = starfish.data.STARmap(use_test_data=True)
stack = experiment['fov_000'].get_image('primary')

###############################################################################
# Look at the max projection of channels and rounds. Ideally, these should form
# fairly coherent spots, indicating that the data are well registered. By
# contrast, if there are patterns whereby pairs of spots are consistently
# present at small shifts, that can indicate systematic registration offsets
# which should be corrected prior to analysis.

ch_r_projection = stack.max_proj(Axes.CH, Axes.ROUND)

f, ax = plt.subplots()
imshow_plane(ch_r_projection, sel={Axes.Z: 15}, ax=ax)

###############################################################################
# It actually looks like there is a small shift approximately the size of a spot
# in the `x = -y` direction for at least one (round, channel) pair (see top left
# corner for most obvious manifestation).
#
# Attempt a translation registration to fix.

from skimage.feature import register_translation
from skimage.transform import SimilarityTransform


def _register_imagestack(target_image, reference_image, upsample_factor):
    target_image = np.squeeze(target_image)
    reference_image = np.squeeze(reference_image)
    shift, error, phasediff = register_translation(
        target_image, reference_image, upsample_factor=upsample_factor)
    return SimilarityTransform(translation=shift)


# identify the locations of all the spots by max projecting over z
projection = stack.max_proj(Axes.CH, Axes.ZPLANE)
reference_image = projection.sel({Axes.ROUND: 1}).xarray

# learn the transformations for each stack
register_imagestack = partial(
    _register_imagestack, reference_image=reference_image, upsample_factor=100
)
transforms = projection.transform(
    register_imagestack, group_by={Axes.ROUND}, n_processes=1)

###############################################################################
# Print the translations.

[t.translation for (t, ind) in transforms]

###############################################################################
# Unfortunately, simple translation registration can't further improve upon this
# problem. To account for this, a small local search will be allowed in
# the spot finding step to match spots across (round, channel) volumes.

###############################################################################
# The first stage of the STARmap pipeline is to align the intensity
# distributions across channels and rounds. Here we calculate a reference
# distribution by sorting each image's intensities in increasing order and
# averaging the ordered intensities across rounds and channels. All (z, y, x)
# volumes from each round and channel are quantile normalized against this
# reference.

mh = starfish.image.Filter.MatchHistograms({Axes.CH, Axes.ROUND})
stack = mh.run(stack, in_place=True, verbose=True, n_processes=8)

###############################################################################
# Finally, a local blob detector that finds spots in each (z, y, x) volume
# separately is applied. The user selects an "anchor round" and spots found in
# all channels of that round are used to seed a local search across other rounds
# and channels. The closest spot is selected, and any spots outside the search
# radius (here 10 pixels) is discarded.
#
# The Spot finder returns an IntensityTable containing all spots from round
# zero. Note that many of the spots do _not_ identify spots in other rounds and
# channels and will therefore fail decoding. Because of the stringency built
# into the STARmap codebook, it is OK to be relatively permissive with the spot
# finding parameters for this assay.

lsbd = starfish.spots.SpotFinder.LocalSearchBlobDetector(
    min_sigma=1,
    max_sigma=8,
    num_sigma=10,
    threshold=np.percentile(np.ravel(stack.xarray.values), 95),
    exclude_border=2,
    anchor_round=0,
    search_radius=10,
)
intensities = lsbd.run(stack, n_processes=8)

# This viewer call displays all detected spots, regardless of whether or not
# they decode.
# viewer = starfish.display(
#     stack, intensities, radius_multiplier=0.1, mask_intensities=0.01
# )

###############################################################################
# Next, spots are decoded, and only spots that pass the decoding stage are
# displayed.

decoded = experiment.codebook.decode_per_round_max(intensities.fillna(0))
decode_mask = decoded['target'] != 'nan'

viewer = starfish.display(
    stack, decoded[decode_mask], radius_multiplier=0.1, mask_intensities=0.1
)