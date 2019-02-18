#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"hide_input": false, "kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}, "toc": {"nav_menu": {}, "number_sections": true, "sideBar": true, "skip_h1_title": false, "toc_cell": false, "toc_position": {}, "toc_section_display": "block", "toc_window_display": false}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Reproduce Published results with Starfish
#
#This notebook walks through a workflow that reproduces a MERFISH result for one field of view using the starfish package.
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: END code

# EPY: START code
# EPY: ESCAPE %matplotlib inline

import pprint
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from showit import image as show_image

import starfish.display
from starfish import data, FieldOfView
from starfish.types import Features, Axes
# EPY: END code

# EPY: START code
# load the data from cloudfront
use_test_data = os.getenv("USE_TEST_DATA") is not None
experiment = data.MERFISH(use_test_data=use_test_data)
# EPY: END code

# EPY: START markdown
#Individual imaging rounds and channels can also be visualized
# EPY: END markdown

# EPY: START code
primary_image = experiment.fov()[FieldOfView.PRIMARY_IMAGES]
# EPY: END code

# EPY: START code
# Display the data
starfish.display.stack(primary_image)
# EPY: END code

# EPY: START markdown
### Show input file format that specifies how the tiff stack is organized
#
#The stack contains multiple images corresponding to the channel and imaging rounds. MERFISH builds a 16 bit barcode from 8 imaging rounds, each of which measures two channels that correspond to contiguous (but not necessarily consistently ordered) bits of the barcode.
#
#The MERFISH computational pipeline also constructs a scalar that corrects for intensity differences across each of the 16 images, e.g., one scale factor per bit position.
#
#The stacks in this example are pre-registered using fiduciary beads.
# EPY: END markdown

# EPY: START code
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(experiment._src_doc)
# EPY: END code

# EPY: START markdown
### Visualize codebook
# EPY: END markdown

# EPY: START markdown
#The MERFISH codebook maps each barcode to a gene (or blank) feature. The codes in the MERFISH codebook are constructed from a 4-hamming error correcting code with exactly 4 "on" bits per barcode
# EPY: END markdown

# EPY: START code
experiment.codebook
# EPY: END code

# EPY: START markdown
### Filter and scale raw data before decoding
# EPY: END markdown

# EPY: START markdown
#Begin filtering with a high pass filter to remove background signal.
# EPY: END markdown

# EPY: START code
from starfish.image import Filter
ghp = Filter.GaussianHighPass(sigma=3)
high_passed = ghp.run(primary_image, verbose=True, in_place=False)
# EPY: END code

# EPY: START markdown
#The below algorithm deconvolves out the point spread function introduced by the microcope and is specifically designed for this use case. The number of iterations is an important parameter that needs careful optimization.
# EPY: END markdown

# EPY: START code
dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, clip=True)
deconvolved = dpsf.run(high_passed, verbose=True, in_place=False)
# EPY: END code

# EPY: START markdown
#Recall that the image is pre-registered, as stated above. Despite this, individual RNA molecules may still not be perfectly aligned across imaging rounds. This is crucial in order to read out a measure of the itended barcode (across imaging rounds) in order to map it to the codebook. To solve for potential mis-alignment, the images can be blurred with a 1-pixel Gaussian kernel. The risk here is that this will obfuscate signals from nearby molecules.
#
#A local search in pixel space across imaging rounds can also solve this.
# EPY: END markdown

# EPY: START code
glp = Filter.GaussianLowPass(sigma=1)
low_passed = glp.run(deconvolved, in_place=False, verbose=True)
# EPY: END code

# EPY: START markdown
#Use MERFISH-calculated size factors to scale the channels across the imaging rounds and visualize the resulting filtered and scaled images. Right now we have to extract this information from the metadata and apply this transformation manually.
# EPY: END markdown

# EPY: START code
if use_test_data:
    scale_factors = {
        (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
        for t in experiment.extras['scale_factors']
    }
else:
    scale_factors = {
        (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
        for index, t in primary_image.tile_metadata.iterrows()
    }
# EPY: END code

# EPY: START code
# this is a scaling method. It would be great to use image.apply here. It's possible, but we need to expose H & C to
# at least we can do it with get_slice and set_slice right now.
from copy import deepcopy
scaled_image = deepcopy(low_passed)

for selector in primary_image._iter_axes():
    data = scaled_image.get_slice(selector)[0]
    scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
    scaled_image.set_slice(selector, scaled, [Axes.ZPLANE])
# EPY: END code

# EPY: START markdown
### Use spot-detector to create 'encoder' table  for standardized input  to decoder
#
#Each pipeline exposes a spot detector, and this spot detector translates the filtered image into an encoded table by detecting spots. The table contains the spot_id, the corresponding intensity (v) and the channel (c), imaging round (r) of each spot.
#
#The MERFISH pipeline merges these two steps together by finding pixel-based features, and then later collapsing these into spots and filtering out undesirable (non-spot) features.
#
#Therefore, no encoder table is generated, but a robust SpotAttribute and DecodedTable are both produced:
# EPY: END markdown

# EPY: START markdown
### Decode
#
#Each assay type also exposes a decoder. A decoder translates each spot (spot_id) in the encoded table into a gene that matches a barcode in the codebook. The goal is to decode and output a quality score, per spot, that describes the confidence in the decoding. Recall that in the MERFISH pipeline, each 'spot' is actually a 16 dimensional vector, one per pixel in the image. From here on, we will refer to these as pixel vectors. Once these pixel vectors are decoded into gene values, contiguous pixels that are decoded to the same gene are labeled as 'spots' via a connected components labeler. We shall refer to the latter as spots.
#
#There are hard and soft decodings -- hard decoding is just looking for the max value in the code book. Soft decoding, by contrast, finds the closest code by distance in intensity. Because different assays each have their own intensities and error modes, we leave decoders as user-defined functions.
#
#For MERFISH, which uses soft decoding, there are several parameters which are important to determining the result of the decoding method:
#
#### Distance threshold
#In MERFISH, each pixel vector is a 16d vector that we want to map onto a barcode via minimum euclidean distance. Each barcode in the codebook, and each pixel vector is first mapped to the unit sphere by L2 normalization. As such, the maximum distance between a pixel vector and the nearest single-bit error barcode is 0.5176. As such, the decoder only accepts pixel vectors that are below this distance for assignment to a codeword in the codebook.
#
#### Magnitude threshold
#This is a signal floor for decoding. Pixel vectors with an L2 norm below this floor are not considered for decoding.
#
#### Area threshold
#Contiguous pixels that decode to the same gene are called as spots via connected components labeling. The minimum area of these spots are set by this parameter. The intuition is that pixel vectors, that pass the distance and magnitude thresholds, shold probably not be trusted as genes as the mRNA transcript would be too small for them to be real. This parameter can be set based on microscope resolution and signal amplification strategy.
#
#### Crop size
#The crop size crops the image by a number of pixels large enough to eliminate parts of the image that suffer from boundary effects from both signal aquisition (e.g., FOV overlap) and image processing. Here this value is 40.
#
#Given these three thresholds, for each pixel vector, the decoder picks the closest code (minimum distance) that satisfies each of the above thresholds, where the distance is calculated between the code and a normalized intensity vector and throws away subsequent spots that are too small.
# EPY: END markdown

# EPY: START code
# TODO this crop should be (x, y) = (40, 40) but it was getting eaten by kwargs
from starfish.spots import PixelSpotDecoder
psd = PixelSpotDecoder.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean',
    distance_threshold=0.5176,
    magnitude_threshold=1.77e-5,
    min_area=2,
    max_area=np.inf,
    norm_order=2,
    crop_z=0,
    crop_y=0,
    crop_x=0
)

initial_spot_intensities, prop_results = psd.run(scaled_image)

spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]
# EPY: END code

# EPY: START markdown
### Compare to results from paper
#
#The below plot aggregates gene copy number across single cells in the field of view and compares the results to the published intensities in the MERFISH paper.
#
#To make this match perfectly, run deconvolution 15 times instead of 14. As presented below, STARFISH displays a lower detection rate.
# EPY: END markdown

# EPY: START code
bench = pd.read_csv('https://d2nhj9g34unfro.cloudfront.net/MERFISH/benchmark_results.csv',
                    dtype = {'barcode':object})

benchmark_counts = bench.groupby('gene')['gene'].count()
genes, counts = np.unique(spot_intensities[Features.AXIS][Features.TARGET], return_counts=True)
result_counts = pd.Series(counts, index=genes)

tmp = pd.concat([result_counts, benchmark_counts], join='inner', axis=1).values

r = np.corrcoef(tmp[:, 1], tmp[:, 0])[0, 1]
x = np.linspace(50, 2000)
f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(tmp[:, 1], tmp[:, 0], 50, zorder=2)
ax.plot(x, x, '-k', zorder=1)

plt.xlabel('Gene copy number Benchmark')
plt.ylabel('Gene copy number Starfish')
plt.xscale('log')
plt.yscale('log')
plt.title(f'r = {r}');
# EPY: END code

# EPY: START markdown
### Visualize results
#
#This image applies a pseudo-color to each gene channel to visualize the position and size of all called spots in a subset of the test image
# EPY: END markdown

# EPY: START code
from scipy.stats import scoreatpercentile
import warnings

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    area_lookup = lambda x: 0 if x == 0 else prop_results.region_properties[x - 1].area
    vfunc = np.vectorize(area_lookup)
    mask = np.squeeze(vfunc(prop_results.label_image))
    show_image(np.squeeze(prop_results.decoded_image)*(mask > 2), cmap='nipy_spectral', ax=ax1)
    ax1.axes.set_axis_off()

    mp = scaled_image.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    mp_numpy = mp._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    clim = scoreatpercentile(mp_numpy, [0.5, 99.5])
    show_image(mp_numpy, clim=clim, ax=ax2)

    f.tight_layout()
# EPY: END code
