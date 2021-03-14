"""
.. _merfish_example:

Reproduce published MERFISH results with starfish
=================================================

Multiplexed Error Robust Fish (MERFISH) is an image based transcriptomics technique that can
spatially resolve hundreds to thousands of RNA species and their expression levels in-situ. The
protocol and data analysis are described in this `publication`_. This notebook walks through how
to use starfish to process the raw images from a MERFISH experiment into a spatially resolved
cell by gene expression matrix. We verify that Starfish can accurately reproduce the results from
the current Matlab based MERFISH `pipeline`_.

.. _publication: https://science.sciencemag.org/content/348/6233/aaa6090
.. _pipeline: https://github.com/ZhuangLab/MERFISH_analysis
"""

from IPython import get_ipython
import matplotlib
import matplotlib.pyplot as plt

# equivalent to %gui qt and %matplotlib inline
ipython = get_ipython()
ipython.magic("gui qt5")
ipython.magic("matplotlib inline")

matplotlib.rcParams["figure.dpi"] = 150

###################################################################################################
# Load Data
# ---------
# The example data here correspond to DARTFISHv1 2017. The data represent human brain tissue from
# the human occipital cortex from 1 field of view (FOV) of larger experiment. The data from one
# field of view correspond to 18 images from 6 imaging rounds (r) 3 color channels (c) and 1 z-plane
# (z). Each image is 988x988 (y,x)

import pprint
from starfish import data
from starfish import FieldOfView

experiment = data.MERFISH(use_test_data=False)

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(experiment._src_doc)

# note the structure of the 5D tensor containing the raw imaging data
imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(imgs)

###################################################################################################
# Visualize codebook
# ------------------
# The MERFISH codebook maps each barcode to a gene (or blank) feature. The barcodes are 16 bit
# vectors that can be read out, for each pixel, from the 8 rounds and 2 color channels. The codebook
# contains a precise specification of how each of these 16 bit barcode vectors relate to the 5D
# tensor of raw image data.

experiment.codebook

###################################################################################################
# Visualize raw data
# ------------------
# We can view an image from the first round and color channel.


from starfish.types import Axes

single_plane = imgs.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 0})
single_plane = single_plane.xarray.squeeze()
plt.figure(figsize=(7, 7))
plt.imshow(single_plane, cmap='gray')
plt.title('Round: 0, Channel: 0')
plt.axis('off')

###################################################################################################
# Filter and scale raw data before decoding into spatially resolved gene expression
# ---------------------------------------------------------------------------------
# A high pass filter is used to remove background signal, which is typically of a low frequency.
# This serves to remove autoflourescence, thus enhancing the ability to detect the RNA molecules.

from starfish.image import Filter

ghp = Filter.GaussianHighPass(sigma=3)
high_passed = ghp.run(imgs, verbose=True, in_place=False)

###################################################################################################
# The below algorithm deconvolves the point spread function (PSF) introduced by the microscope. The
# goal of deconvolution is to enable the resolution of more spots, especially in high transcript
# density regions of the data. For this assay, the PSF is well approximated by a 2D isotropic
# gaussian with standard deviation (sigma) of 2. This The number of iterations (here 15) is an
# important parameter that needs careful optimization.

from starfish.types import Levels

dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
deconvolved = dpsf.run(high_passed, verbose=True, in_place=False)

###################################################################################################
# The data for this assay are already registered across imaging rounds. Despite this, individual RNA
# molecules may still not be perfectly aligned across imaging rounds. This is crucial in order to
# read out a measure of the intended barcode (across imaging rounds) in order to map it to the
# codebook. To solve for potential mis-alignment, the images can be blurred with a 1-pixel Gaussian
# kernel. The risk here is that this will obfuscate signals from nearby molecules, thus potentially
# working against the deconvolution step previously carried out.

glp = Filter.GaussianLowPass(sigma=1)
low_passed = glp.run(deconvolved, in_place=False, verbose=True)

###################################################################################################
# Image intensities vary across color channels and imaging rounds. We use the author's computed
# scale factors to appropriately scale the data to correct for this variation. Right now we have to
# extract this information from the metadata and apply this transformation manually.

scale_factors = {
    (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
    for t in experiment.extras['scale_factors']
}

# this is a scaling method. It would be great to use image.apply here. It's possible, but we need to expose H & C to
# at least we can do it with get_slice and set_slice right now.
from copy import deepcopy
filtered_imgs = deepcopy(low_passed)

for selector in imgs._iter_axes():
    data = filtered_imgs.get_slice(selector)[0]
    scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
    filtered_imgs.set_slice(selector, scaled, [Axes.ZPLANE])

###################################################################################################
# Visualize processed data
# ------------------------

import numpy as np

single_plane_filtered = filtered_imgs.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 0})
single_plane_filtered = single_plane_filtered.xarray.squeeze()

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(single_plane, cmap='gray', clim=list(np.percentile(single_plane.data, [5, 99])))
plt.axis('off')
plt.title('Original data, Round: 0, Channel: 0')
plt.subplot(122)
plt.imshow(single_plane_filtered, cmap='gray', clim=list(np.percentile(single_plane_filtered.data, [5, 99])))
plt.title('Filtered data, Round: 0, Channel: 0')
plt.axis('off')

###################################################################################################
# Decode the processed data into spatially resolved gene expression profiles
# --------------------------------------------------------------------------
# Here, we decode each pixel value, across all rounds and channels, into the corresponding target
# (gene) it corresponds too. Contiguous pixels that map to the same target gene are called as one
# RNA molecule. Intuitively, pixel vectors are matched to the codebook by computing the euclidean
# distance between the pixel vector and all codewords. The minimal distance gene target is selected
# if it lies within distance_threshold of a code.

from starfish.spots import DetectPixels
from starfish.types import Features

psd = DetectPixels.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean', # distance metric to use for computing distance between a pixel vector and a codeword
    norm_order=2, # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
    distance_threshold=0.5176, # minimum distance between a pixel vector and a codeword for it to be called as a gene
    magnitude_threshold=1.77e-5, # discard any pixel vectors below this magnitude
    min_area=2, # do not call a 'spot' if it's area is below this threshold (measured in pixels)
    max_area=np.inf, # do not call a 'spot' if it's area is above this threshold (measured in pixels)
)

initial_spot_intensities, prop_results = psd.run(filtered_imgs)

spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]

###################################################################################################
# Compare to results from paper
# -----------------------------
# The below plot aggregates gene copy number across single cells in the field of view and compares
# the results to the published counts in the MERFISH paper. Note that Starfish detects a lower
# number of transcripts than the authors' results. This can likely be improved by tweaking the
# parameters of the algorithms above.

import pandas as pd

bench = pd.read_csv('https://d2nhj9g34unfro.cloudfront.net/MERFISH/benchmark_results.csv',
                    dtype={'barcode': object})

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
plt.title(f'r = {r}')

###################################################################################################
# Visualize results
# -----------------
# This image applies a pseudo-color to each gene channel to visualize the position and size of all
# called spots in a subset of the test image.

from showit import image as show_image
from scipy.stats import scoreatpercentile
import warnings

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    area_lookup = lambda x: 0 if x == 0 else prop_results.region_properties[x - 1].area
    vfunc = np.vectorize(area_lookup)
    mask = np.squeeze(vfunc(prop_results.label_image))
    show_image(np.squeeze(prop_results.decoded_image)*(mask > 2), cmap='nipy_spectral', ax=ax1)
    ax1.axes.set_axis_off()

    mp_numpy = filtered_imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")._squeezed_numpy(
        Axes.ROUND, Axes.CH, Axes.ZPLANE)
    clim = scoreatpercentile(mp_numpy, [0.5, 99.5])
    show_image(mp_numpy, clim=clim, ax=ax2)

    f.tight_layout()
