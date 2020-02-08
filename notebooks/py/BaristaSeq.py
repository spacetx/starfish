#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish-CI", "language": "python", "name": "starfish-ci"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.7.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
# EPY: ESCAPE %matplotlib inline
# EPY: END code

# EPY: START markdown
## BaristaSeq
#
#BaristaSeq is an assay that sequences padlock-probe initiated rolling circle
#amplified spots using a one-hot codebook. The publication for this assay can be
#found [here](https://www.ncbi.nlm.nih.gov/pubmed/29190363)
#
#This example processes a single field of view extracted from a tissue slide that
#measures gene expression in mouse primary visual cortex.
# EPY: END markdown

# EPY: START code
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import starfish
import starfish.data
from starfish import FieldOfView
from starfish.types import Axes, Levels
from starfish.util.plot import (
    imshow_plane, intensity_histogram, overlay_spot_calls
)

matplotlib.rcParams["figure.dpi"] = 150
# EPY: END code

# EPY: START markdown
#Load Data
#---------
#Import starfish and extract a single field of view.
# EPY: END markdown

# EPY: START code
exp = starfish.data.BaristaSeq(use_test_data=False)

nissl = exp.fov().get_image('nuclei')
img = exp.fov().get_image(FieldOfView.PRIMARY_IMAGES)
# EPY: END code

# EPY: START markdown
#starfish data are 5-dimensional, but to demonstrate what they look like in a
#non-interactive fashion, it's best to visualize the data in 2-d. There are
#better ways to look at these data using the `starfish.display`
#method, which allows the user to page through each axis of the tensor
# EPY: END markdown

# EPY: START code
# for this vignette, we'll pick one plane and track it through the processing
# steps
plane_selector = {Axes.CH: 0, Axes.ROUND: 0, Axes.ZPLANE: 8}

f, (ax1, ax2) = plt.subplots(ncols=2)
imshow_plane(img, sel=plane_selector, ax=ax1, title="primary image")
imshow_plane(nissl, sel=plane_selector, ax=ax2, title="nissl image")
# EPY: END code

# EPY: START markdown
#Register the data
#-----------------
#The first step in BaristaSeq is to do some rough registration. For this data,
#the rough registration has been done for us by the authors, so it is omitted
#from this notebook.
# EPY: END markdown

# EPY: START markdown
#Project into 2D
#---------------
#BaristaSeq is typically processed in 2d. Starfish allows users to reduce data using arbitrary
#methods via `starfish.image.Filter.Reduce`.  Here we max project Z for both the nissl images and
#the primary images.
# EPY: END markdown

# EPY: START code
from starfish.image import Filter
from starfish.types import FunctionSource
max_projector = Filter.Reduce((Axes.ZPLANE,), func=FunctionSource.np("max"))
z_projected_image = max_projector.run(img)
z_projected_nissl = max_projector.run(nissl)

# show the projected data
f, (ax1, ax2) = plt.subplots(ncols=2)
imshow_plane(z_projected_image, sel={Axes.CH: 0, Axes.ROUND: 0}, ax=ax1, title="primary image")
imshow_plane(z_projected_nissl, sel={Axes.CH: 0, Axes.ROUND: 0}, title="nissl image")
# EPY: END code

# EPY: START markdown
#Correct Channel Misalignment
#----------------------------
#There is a slight miss-alignment of the C channel in the microscope used to
#acquire the data. This has been corrected for this data, but here is how it
#could be transformed using python code for future datasets.
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
#Remove Registration Artefacts
#-----------------------------
#There are some minor registration errors along the pixels for which y < 100
#and x < 50. Those pixels are dropped from this analysis
# EPY: END markdown

# EPY: START code
registration_corrected: starfish.ImageStack = z_projected_image.sel(
    {Axes.Y: (100, -1), Axes.X: (50, -1)}
)
# EPY: END code

# EPY: START markdown
#Correct for bleed-through from Illumina SBS reagents
#----------------------------------------------------
#The following matrix contains bleed correction factors for Illumina
#sequencing-by-synthesis reagents. Starfish provides a LinearUnmixing method
#that will unmix the fluorescence intensities
# EPY: END markdown

# EPY: START code
data = np.array(
    [[ 1.  , -0.05,  0.  ,  0.  ],
     [-0.35,  1.  ,  0.  ,  0.  ],
     [ 0.  , -0.02,  1.  , -0.84],
     [ 0.  ,  0.  , -0.05,  1.  ]]
)
rows = pd.Index(np.arange(4), name='bleed_from')
cols = pd.Index(np.arange(4), name='bleed_to')
unmixing_coeff = pd.DataFrame(data, rows, cols)

lum = starfish.image.Filter.LinearUnmixing(unmixing_coeff)
bleed_corrected = lum.run(registration_corrected, in_place=False)
# EPY: END code

# EPY: START markdown
#the matrix shows that (zero-based!) channel 2 bleeds particularly heavily into
#channel 3. To demonstrate the effect of unmixing, we'll plot channels 2 and 3
#of round 0 before and after unmixing.
#
#Channel 2 should look relative unchanged, as it only receives a bleed through
#of 5% of channel 3. However, Channel 3 should look dramatically sparser after
#spots from Channel 2 have been subtracted
# EPY: END markdown

# EPY: START code
# TODO ambrosejcarr fix this.
ch2_r0 = {Axes.CH: 2, Axes.ROUND: 0, Axes.X: (500, 700), Axes.Y: (500, 700)}
ch3_r0 = {Axes.CH: 3, Axes.ROUND: 0, Axes.X: (500, 700), Axes.Y: (500, 700)}
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
imshow_plane(
    registration_corrected,
    sel=ch2_r0, ax=ax1, title="Channel 2\nBefore Unmixing"
)
imshow_plane(
    registration_corrected,
    sel=ch3_r0, ax=ax2, title="Channel 3\nBefore Unmixing"
)
imshow_plane(
    bleed_corrected,
    sel=ch2_r0, ax=ax3, title="Channel 2\nAfter Unmixing"
)
imshow_plane(
    bleed_corrected,
    sel=ch3_r0, ax=ax4, title="Channel 3\nAfter Unmixing"
)
f.tight_layout()
# EPY: END code

# EPY: START markdown
#Remove image background
#-----------------------
#To remove image background, BaristaSeq uses a White Tophat filter, which
#measures the background with a rolling disk morphological element and
#subtracts it from the image.
# EPY: END markdown

# EPY: START code
from skimage.morphology import opening, dilation, disk
from functools import partial

# calculate the background
opening = partial(opening, selem=disk(5))

background = bleed_corrected.apply(
    opening,
    group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False
)

wth = starfish.image.Filter.WhiteTophat(masking_radius=5)
background_corrected = wth.run(bleed_corrected, in_place=False)

f, (ax1, ax2, ax3) = plt.subplots(ncols=3)
selector = {Axes.CH: 0, Axes.ROUND: 0, Axes.X: (500, 700), Axes.Y: (500, 700)}
imshow_plane(bleed_corrected, sel=selector, ax=ax1, title="template\nimage")
imshow_plane(background, sel=selector, ax=ax2, title="background")
imshow_plane(
    background_corrected, sel=selector, ax=ax3, title="background\ncorrected"
)
f.tight_layout()
# EPY: END code

# EPY: START markdown
#Scale images to equalize spot intensities across channels
#---------------------------------------------------------
#The number of peaks are not uniform across rounds and channels,
#which prevents histogram matching across channels. Instead, a percentile value
#is identified and set as the maximum across channels, and the dynamic range is
#extended to equalize the channel intensities. We first demonatrate what
#scaling by the max value does.
# EPY: END markdown

# EPY: START code
sbp = starfish.image.Filter.Clip(p_max=100, level_method=Levels.SCALE_BY_CHUNK)
scaled = sbp.run(background_corrected, n_processes=1, in_place=False)
# EPY: END code

# EPY: START markdown
#The easiest way to visualize this is to calculate the intensity histograms
#before and after this scaling and plot their log-transformed values. This
#should see that the histograms are better aligned in terms of intensities.
#It gets most of what we want, but the histograms are still slightly shifted;
#a result of high-value outliers.
# EPY: END markdown

# EPY: START code
def plot_scaling_result(
    template: starfish.ImageStack, scaled: starfish.ImageStack
):
    f, (before, after) = plt.subplots(ncols=4, nrows=2)
    for channel, ax in enumerate(before):
        title = f'Before scaling\nChannel {channel}'
        intensity_histogram(
            template, sel={Axes.CH: channel, Axes.ROUND: 0}, ax=ax, title=title,
            log=True, bins=50,
        )
        ax.set_xlim(0, 0.007)
    for channel, ax in enumerate(after):
        title = f'After scaling\nChannel {channel}'
        intensity_histogram(
            scaled, sel={Axes.CH: channel, Axes.ROUND: 0}, ax=ax, title=title,
            log=True, bins=50,
        )
    f.tight_layout()
    return f

f = plot_scaling_result(background_corrected, scaled)
# EPY: END code

# EPY: START markdown
#We repeat this scaling by the 99.8th percentile value, which does a better job
#of equalizing the intensity distributions.
#
#It should also be visible that exactly 0.2% of values take on the max value
#of 1. This is a result of setting any value above the 99.8th percentile to 1,
#and is a trade-off made to eliminate large-value outliers.
# EPY: END markdown

# EPY: START code
sbp = starfish.image.Filter.Clip(p_max=99.8, level_method=Levels.SCALE_BY_CHUNK)
scaled = sbp.run(background_corrected, n_processes=1, in_place=False)

f = plot_scaling_result(background_corrected, scaled)
# EPY: END code

# EPY: START markdown
### Detect Spots
#We use a pixel spot decoder to identify the gene target for each spot.
# EPY: END markdown

# EPY: START code
psd = starfish.spots.DetectPixels.PixelSpotDecoder(
    codebook=exp.codebook, metric='euclidean', distance_threshold=0.5,
    magnitude_threshold=0.1, min_area=7, max_area=50
)
pixel_decoded, ccdr = psd.run(scaled)
# EPY: END code

# EPY: START markdown
#plot a mask that shows where pixels have decoded to genes.
# EPY: END markdown

# EPY: START code
f, ax = plt.subplots()
ax.imshow(np.squeeze(ccdr.decoded_image), cmap=plt.cm.nipy_spectral)
ax.axis("off")
ax.set_title("Pixel Decoding Results")
# EPY: END code

# EPY: START markdown
#Get the total counts for each gene from each spot detector.
#Do the below values make sense for this tissue and this probeset?
# EPY: END markdown

# EPY: START code
pixel_decoded_gene_counts = pd.Series(
    *np.unique(pixel_decoded['target'], return_counts=True)[::-1]
)

print(pixel_decoded_gene_counts.sort_values(ascending=False)[:20])
# EPY: END code
