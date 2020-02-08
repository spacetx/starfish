#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
## Starfish SeqFISH Work-in-progress Processing Example
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %gui qt

import os
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import skimage.filters
import skimage.morphology
from skimage.transform import SimilarityTransform, warp
from tqdm import tqdm

import starfish
import starfish.data
from starfish.types import Axes, Levels, TraceBuildingStrategies
# EPY: END code

# EPY: START markdown
#Select data for a single field of view.
# EPY: END markdown

# EPY: START code
exp = starfish.data.SeqFISH(use_test_data=True)
# EPY: END code

# EPY: START code
img = exp['fov_000'].get_image('primary')
# EPY: END code

# EPY: START markdown
#The first step in SeqFISH is to do some rough registration. For this data, the rough registration has been done for us by the authors, so it is omitted from this notebook.
# EPY: END markdown

# EPY: START markdown
### Remove image background
# EPY: END markdown

# EPY: START markdown
#To remove image background, use a White Tophat filter, which measures the background with a rolling disk morphological element and subtracts it from the image.
# EPY: END markdown

# EPY: START code
from skimage.morphology import opening, dilation, disk
from functools import partial
# EPY: END code

# EPY: START markdown
#If desired, the background that is being subtracted can be visualized
# EPY: END markdown

# EPY: START code
opening = partial(opening, selem=disk(3))

background = img.apply(
    opening,
    group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False
)

starfish.display(background)
# EPY: END code

# EPY: START code
wth = starfish.image.Filter.WhiteTophat(masking_radius=3)
background_corrected = wth.run(img, in_place=False)
starfish.display(background_corrected)
# EPY: END code

# EPY: START markdown
### Scale images to equalize spot intensities across channels
#
#The number of peaks are not uniform across rounds and channels, which prevents histogram matching across channels. Instead, a percentile value is identified and set as the maximum across channels, and the dynamic range is extended to equalize the channel intensities
# EPY: END markdown

# EPY: START code
clip = starfish.image.Filter.Clip(p_max=99.9, is_volume=True, level_method=Levels.SCALE_BY_CHUNK)
scaled = clip.run(background_corrected, in_place=False)
# EPY: END code

# EPY: START code
starfish.display(scaled)
# EPY: END code

# EPY: START markdown
### Remove residual background
#
#The background is fairly uniformly present below intensity=0.5. However, starfish's clip method currently only supports percentiles. To solve this problem, the intensities can be directly edited in the underlying numpy array.
# EPY: END markdown

# EPY: START code
from copy import deepcopy
clipped = deepcopy(scaled)
clipped.xarray.values[clipped.xarray.values < 0.7] = 0
# EPY: END code

# EPY: START code
starfish.display(clipped)
# EPY: END code

# EPY: START markdown
### Detect Spots
#
#Detect spots with a local search blob detector that identifies spots in all rounds and channels and matches them using a local search method. The local search starts in an anchor channel (default ch=1) and identifies the nearest spot in all subsequent imaging rounds.
# EPY: END markdown

# EPY: START code
threshold = 0.5

bd = starfish.spots.FindSpots.BlobDetector(
    min_sigma=(1.5, 1.5, 1.5),
    max_sigma=(8, 8, 8),
    num_sigma=10,
    threshold=threshold)

spots = bd.run(clipped)
decoder = starfish.spots.DecodeSpots.PerRoundMaxChannel(
    codebook=exp.codebook,
    search_radius=7,
    trace_building_strategy=TraceBuildingStrategies.NEAREST_NEIGHBOR)

decoded = decoder.run(spots=spots)
# EPY: END code

# EPY: START code
starfish.display(clipped, decoded)
# EPY: END code

# EPY: START markdown
#Based on visual inspection, it looks like the spot correspondence across rounds isn't being detected well. Try the PixelSpotDecoder.
# EPY: END markdown

# EPY: START code
glp = starfish.image.Filter.GaussianLowPass(sigma=(0.3, 1, 1), is_volume=True)
blurred = glp.run(clipped)
# EPY: END code

# EPY: START code
psd = starfish.spots.DetectPixels.PixelSpotDecoder(
    codebook=exp.codebook, metric='euclidean', distance_threshold=0.5,
    magnitude_threshold=0.1, min_area=7, max_area=50,
)
pixel_decoded, ccdr = psd.run(blurred)
# EPY: END code

# EPY: START code
import matplotlib.pyplot as plt
# EPY: END code

# EPY: START code
# look at the label image in napari
label_image = starfish.ImageStack.from_numpy(np.reshape(ccdr.decoded_image, (1, 1, 29, 280, 280)))
starfish.display(label_image)
# EPY: END code

# EPY: START markdown
#Compare the number of spots being detected by the two spot finders
# EPY: END markdown

# EPY: START code
print("pixel_decoder spots detected", int(np.sum(pixel_decoded['target'] != 'nan')))
print("local search spot detector spots detected", int(np.sum(decoded['target'] != 'nan')))
# EPY: END code

# EPY: START markdown
#Report the correlation between the two methods
# EPY: END markdown

# EPY: START code
from scipy.stats import pearsonr

# get the total counts for each gene from each spot detector
pixel_decoded_gene_counts = pd.Series(*np.unique(pixel_decoded['target'], return_counts=True)[::-1])
decoded_gene_counts = pd.Series(*np.unique(decoded['target'], return_counts=True)[::-1])

# get the genes that are detected by both spot finders
codetected = pixel_decoded_gene_counts.index.intersection(decoded_gene_counts.index).drop('nan')

# report the correlation
pearsonr(pixel_decoded_gene_counts[codetected], decoded_gene_counts[codetected])
# EPY: END code
