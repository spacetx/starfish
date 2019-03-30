#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Load Data
# EPY: END markdown

# EPY: START code
import os
from itertools import product
from functools import partial
from typing import Tuple

import numpy as np

import starfish
import starfish.data
from starfish.types import Axes
# EPY: END code

# EPY: START code
# EPY: ESCAPE %gui qt
import starfish.display
# EPY: END code

# EPY: START code
experiment = starfish.data.STARmap(use_test_data=True)
stack = experiment['fov_000'].get_image('primary')
# EPY: END code

# EPY: START code
# look at the channel/round projection
ch_r_projection = stack.max_proj(Axes.CH, Axes.ROUND)
# EPY: END code

# EPY: START code
starfish.display.stack(ch_r_projection)
# EPY: END code

# EPY: START markdown
#It actually looks like there is a small shift approximately the size of a spot in the `x = -y` direction for at least one (round, channel) pair (see top left corner for most obvious manifestation).
#
#Attempt a translation registration to fix. 
# EPY: END markdown

# EPY: START code
# Starmap only requires translation. Verify that things are registered with a quick 
# similarity registration. 

from skimage.feature import register_translation
from skimage.transform import warp
from skimage.transform import SimilarityTransform

def _register_imagestack(target_image, reference_image, upsample_factor=5):
    target_image = np.squeeze(target_image)
    reference_image = np.squeeze(reference_image)
    shift, error, phasediff = register_translation(target_image, reference_image, upsample_factor=1)
    return SimilarityTransform(translation=shift)

# identify the locations of all the spots by max projecting over z
projection = stack.max_proj(Axes.CH, Axes.ZPLANE)
reference_image = projection.sel({Axes.ROUND: 1}).xarray

# learn the transformations for each stack
register_imagestack = partial(
    _register_imagestack, reference_image=reference_image, upsample_factor=5
)
transforms = projection.transform(register_imagestack, group_by={Axes.ROUND}, n_processes=1)
# EPY: END code

# EPY: START code
[t.translation for (t, ind) in transforms]
# EPY: END code

# EPY: START markdown
#Unfortunately, simple translation registration can't improve upon this problem significantly. To account for this, a small local search will be allowed in the spot finding step to match spots across (round, channel) volumes.
# EPY: END markdown

# EPY: START markdown
#The first stage of the STARmap pipeline is to align the intensity distributions across channels and rounds. Here we calculate a reference distribution by sorting each image's intensities in increasing order and averaging the ordered intensities across rounds and channels. All (z, y, x) volumes from each round and channel are quantile normalized against this reference. 
# EPY: END markdown

# EPY: START code
mh = starfish.image.Filter.MatchHistograms({Axes.CH, Axes.ROUND})
stack = mh.run(stack, in_place=True, verbose=True, n_processes=8)
# EPY: END code

# EPY: START markdown
#Finally, a local blob detector that finds spots in each (z, y, x) volume separately is applied. The user selects an "anchor round" and spots found in all channels of that round are used to seed a local search across other rounds and channels. The closest spot is selected, and any spots outside the search radius (here 10 pixels) is discarded.
#
#The Spot finder returns an IntensityTable containing all spots from round zero. Note that many of the spots do _not_ identify spots in other rounds and channels and will therefore fail decoding. Because of the stringency built into the STARmap codebook, it is OK to be relatively permissive with the spot finding parameters for this assay.
# EPY: END markdown

# EPY: START code
lsbd = starfish.spots._detector.local_search_blob_detector.LocalSearchBlobDetector(
    min_sigma=1,
    max_sigma=8,
    num_sigma=10,
    threshold=np.percentile(np.ravel(substack.xarray.values), 95),
    exclude_border=2,
    anchor_round=0,
    search_radius=10,
)
intensities = lsbd.run(stack, n_processes=8)
# EPY: END code

# EPY: START markdown
#This viewer call displays all detected spots, regardless of whether or not they decode. 
# EPY: END markdown

# EPY: START code
viewer = starfish.display.stack(stack, intensities, radius_multiplier=0.1, mask_intensities=0.01)
# EPY: END code

# EPY: START markdown
#Next, spots are decoded, and only spots that pass the decoding stage are displayed. 
# EPY: END markdown

# EPY: START code
decoded = experiment.codebook.decode_per_round_max(intensities.fillna(0))
decode_mask = decoded['target'] != 'nan'
# EPY: END code

# EPY: START code
viewer = starfish.display.stack(stack, decoded[decode_mask], radius_multiplier=0.1, mask_intensities=0.1)
# EPY: END code
