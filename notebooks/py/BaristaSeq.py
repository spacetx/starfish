#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
## Starfish BaristaSeq Processing Example
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
from starfish.spots import SpotFinder
from starfish.types import Axes
# EPY: END code

# EPY: START markdown
#BaristaSeq is an assay that sequences padlock-probe initiated rolling circle amplified spots using a one-hot codebook. The publication for this assay can be found [here](https://www.ncbi.nlm.nih.gov/pubmed/29190363).
#
#here we select data for a single field of view. 
# EPY: END markdown

# EPY: START code
experiment_json = "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20190319/baristaseq/experiment.json"
exp = starfish.Experiment.from_json(experiment_json)

nissl = exp['fov_000'].get_image('dots')
img = exp['fov_000'].get_image('primary')
# EPY: END code

# EPY: START markdown
#The first step in BaristaSeq is to do some rough registration. For this data, the rough registration has been done for us by the authors, so it is omitted from this notebook.
# EPY: END markdown

# EPY: START markdown
### Project into 2D
#
#First, project the z-plane to do analysis of BaristaSeq in 2-d. 
# EPY: END markdown

# EPY: START code
z_projected_image = img.max_proj(Axes.ZPLANE)
z_projected_nissl = nissl.max_proj(Axes.ZPLANE)
# EPY: END code

# EPY: START markdown
### Correct Channel Misalignment
#
#There is a slight miss-alignment of the C channel in the microscope used to process the data. This has been corrected for this data, but here is how it could be transformed using python code for future datasets.
# EPY: END markdown

# EPY: START code
# from skimage.feature import register_translation
# from skimage.transform import warp
# from skimage.transform import SimilarityTransform
# from functools import partial

# # Define the translation
# transform = SimilarityTransform(translation=(1.9, -0.4))

# # C is channel 0
# channels = (0,)

# # The channel should be transformed in all rounds
# rounds = np.arange(img.num_rounds)

# # apply the transformation in place
# slice_indices = product(channels, rounds)
# for ch, round_, in slice_indices:
#     selector = {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: 0}
#     tile = z_projected_image.get_slice(selector)[0]
#     transformed = warp(tile, transform)
#     z_projected_image.set_slice(
#         selector=selector,
#         data=transformed.astype(np.float32),
#     )
# EPY: END code

# EPY: START markdown
### Remove Registration Artefacts
#
#There are some minor registration errors along the pixels for which y < 100 and x < 50. Those pixels are dropped from this analysis
# EPY: END markdown

# EPY: START code
registration_corrected = z_projected_image.sel({Axes.Y: (100, -1), Axes.X: (50, -1)})
# EPY: END code

# EPY: START markdown
### Correct for bleed-through from Illumina SBS reagents
#
#The following matrix contains bleed correction factors for Illumina sequencing-by-synthesis reagents. Starfish provides a LinearUnmixing method that will unmix the fluorescence intensities
# EPY: END markdown

# EPY: START code
data = np.array(
    [[0.  , 0.05, 0.  , 0.  ],
     [0.35, 0.  , 0.  , 0.  ],
     [0.  , 0.02, 0.  , 0.84],
     [0.  , 0.  , 0.05, 0.  ]]
)
rows = pd.Index(np.arange(4), name='bleed_from')
cols = pd.Index(np.arange(4), name='bleed_to')
unmixing_coeff = pd.DataFrame(data, rows, cols)

# show results
unmixing_coeff
# EPY: END code

# EPY: START code
lum = starfish.image._filter.linear_unmixing.LinearUnmixing(unmixing_coeff)
bleed_corrected = lum.run(registration_corrected)
# EPY: END code

# EPY: START markdown
### Remove image background
# EPY: END markdown

# EPY: START markdown
#To remove image background, BaristaSeq uses a White Tophat filter, which measures the background with a rolling disk morphological element and subtracts it from the image. 
# EPY: END markdown

# EPY: START code
from skimage.morphology import opening, dilation, disk
from functools import partial
# EPY: END code

# EPY: START markdown
#If desired, the background that is being subtracted can be visualized
# EPY: END markdown

# EPY: START code
# opening = partial(opening, selem=disk(5))

# background = bleed_corrected.apply(
#     opening,
#     group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False
# )

# starfish.display(background)
# EPY: END code

# EPY: START code
wth = starfish.image.Filter.WhiteTophat(masking_radius=5)
background_corrected = wth.run(bleed_corrected, in_place=False)
# EPY: END code

# EPY: START markdown
### Scale images to equalize spot intensities across channels
#
#The number of peaks are not uniform across rounds and channels, which prevents histogram matching across channels. Instead, a percentile value is identified and set as the maximum across channels, and the dynamic range is extended to equalize the channel intensities
# EPY: END markdown

# EPY: START code
sbp = starfish.image.Filter.ScaleByPercentile(p=99.5)
scaled = sbp.run(background_corrected, n_processes=1, in_place=False)
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
clipped.xarray.values[clipped.xarray.values < 0.5] = 0
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

lsbd = starfish.spots._detector.local_search_blob_detector.LocalSearchBlobDetector(
    min_sigma=(0.5, 0.5, 0.5),
    max_sigma=(8, 8, 8),
    num_sigma=10,
    threshold=threshold,
    search_radius=7
)
intensities = lsbd.run(clipped)
decoded = exp.codebook.decode_per_round_max(intensities.fillna(0))
# EPY: END code

# EPY: START code
starfish.display(clipped, intensities)
# EPY: END code

# EPY: START markdown
#Based on visual inspection, it looks like the spot correspondence across rounds isn't being detected well. Try the PixelSpotDecoder.
# EPY: END markdown

# EPY: START code
psd = starfish.spots.PixelSpotDecoder.PixelSpotDecoder(
    codebook=exp.codebook, metric='euclidean', distance_threshold=0.5, 
    magnitude_threshold=0.1, min_area=7, max_area=50
)
pixel_decoded, ccdr = psd.run(clipped)
# EPY: END code

# EPY: START code
label_image = starfish.ImageStack.from_numpy_array(np.reshape(ccdr.label_image, (1, 1, 1, 1092, 862)))
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
codetected = pixel_decoded_gene_counts.index.intersection(decoded_gene_counts.index)

# report the correlation
pearsonr(pixel_decoded_gene_counts[codetected], decoded_gene_counts[codetected])
# EPY: END code

# EPY: START markdown
#The pixel based spot detector looks better upon visual inspection. Do the below values make sense for this tissue and this probeset?? 
# EPY: END markdown

# EPY: START code
pixel_decoded_gene_counts.sort_values()
# EPY: END code

# EPY: START code
exp.codebook[np.where(exp.codebook["target"] == "Ctxn1")]
# EPY: END code

# EPY: START code
exp.codebook[np.where(exp.codebook["target"] == "Ptn")]
# EPY: END code

# EPY: START code
exp.codebook[np.where(exp.codebook["target"] == "Brinp3")]
# EPY: END code

# EPY: START markdown
#Looks like the codebook targets from PixelSpotDecoding don't share much in the way of channel biases across rounds or across codes, which makes me reasonably confident in the decoding result. 
# EPY: END markdown
