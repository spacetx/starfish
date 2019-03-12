#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"hide_input": false, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}, "toc": {"nav_menu": {}, "number_sections": true, "sideBar": true, "skip_h1_title": false, "toc_cell": false, "toc_position": {}, "toc_section_display": "block", "toc_window_display": false}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Reproduce In-situ Sequencing results with Starfish
#
#This notebook walks through a work flow that reproduces an ISS result for one field of view using the starfish package.
#
### Load tiff stack and visualize one field of view
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %matplotlib inline
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from showit import image
import pprint

from starfish import data, FieldOfView
from starfish.types import Features, Axes
# EPY: END code

# EPY: START code
use_test_data = os.getenv("USE_TEST_DATA") is not None
experiment = data.ISS(use_test_data=use_test_data)


# s.image.squeeze() simply converts the 4D tensor H*C*X*Y into a list of len(H*C) image planes for rendering by 'tile'
# EPY: END code

# EPY: START markdown
### Show input file format that specifies how the tiff stack is organized
#
#The stack contains multiple single plane images, one for each color channel, 'c', (columns in above image) and imaging round, 'r', (rows in above image). This protocol assumes that genes are encoded with a length 4 quatenary barcode that can be read out from the images. Each round encodes a position in the codeword. The maximum signal in each color channel (columns in the above image) corresponds to a letter in the codeword. The channels, in order, correspond to the letters: 'T', 'G', 'C', 'A'. The goal is now to process these image data into spatially organized barcodes, e.g., ACTG, which can then be mapped back to a codebook that specifies what gene this codeword corresponds to.
# EPY: END markdown

# EPY: START code
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(experiment._src_doc)
# EPY: END code

# EPY: START markdown
#The flat TIFF files are loaded into a 4-d tensor with dimensions corresponding to imaging round, channel, x, and y. For other volumetric approaches that image the z-plane, this would be a 5-d tensor.
# EPY: END markdown

# EPY: START code
fov = experiment.fov()
primary_image = fov.get_image(FieldOfView.PRIMARY_IMAGES)
dots = fov.get_image('dots')
nuclei = fov.get_image('nuclei')
images = [primary_image, nuclei, dots]
# EPY: END code

# EPY: START code
# round, channel, x, y, z
primary_image.xarray.shape
# EPY: END code

# EPY: START markdown
### Show auxiliary images captured during the experiment
# EPY: END markdown

# EPY: START markdown
#'dots' is a general stain for all possible transcripts. This image should correspond to the maximum projcection of all color channels within a single imaging round. This auxiliary image is useful for registering images from multiple imaging rounds to this reference image. We'll see an example of this further on in the notebook
# EPY: END markdown

# EPY: START code
dots_mp = dots.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
dots_mp_numpy = dots._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
image(dots_mp_numpy)
# EPY: END code

# EPY: START markdown
#Below is a DAPI auxiliary image, which specifically marks nuclei. This is useful cell segmentation later on in the processing.
# EPY: END markdown

# EPY: START code
nuclei_mp = nuclei.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
nuclei_mp_numpy = nuclei_mp._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
image(nuclei_mp_numpy)
# EPY: END code

# EPY: START markdown
### Examine the codebook
# EPY: END markdown

# EPY: START markdown
#Each 4 letter quatenary code (as read out from the 4 imaging rounds and 4 color channels) represents a gene. This relationship is stored in a codebook
# EPY: END markdown

# EPY: START code
experiment.codebook
# EPY: END code

# EPY: START markdown
### Filter and scale raw data
#
#Now apply the white top hat filter to both the spots image and the individual channels. White top had enhances white spots on a black background.
# EPY: END markdown

# EPY: START code
from starfish.image import Filter

# filter raw data
masking_radius = 15
filt = Filter.WhiteTophat(masking_radius, is_volume=False)
for img in images:
    filt.run(img, verbose=True, in_place=True)
# EPY: END code

# EPY: START markdown
### Register data
# EPY: END markdown

# EPY: START markdown
#For each imaging round, the max projection across color channels should look like the dots stain.
#Below, this computes the max projection across the color channels of an imaging round and learns the linear transformation to maps the resulting image onto the dots image.
#
#The Fourier shift registration approach can be thought of as maximizing the cross-correlation of two images.
#
#In the below table, Error is the minimum mean-squared error, and shift reports changes in x and y dimension.
# EPY: END markdown

# EPY: START code
from starfish.image import Registration

registration = Registration.FourierShiftRegistration(
    upsampling=1000,
    reference_stack=dots,
    verbose=True)
registered_image = registration.run(primary_image, in_place=False)
# EPY: END code

# EPY: START markdown
### Use spot-detector to create 'encoder' table  for standardized input  to decoder
# EPY: END markdown

# EPY: START markdown
#Each pipeline exposes an encoder that translates an image into spots with intensities.  This approach uses a Gaussian spot detector.
# EPY: END markdown

# EPY: START code
from starfish.spots import SpotFinder
import warnings

# parameters to define the allowable gaussian sizes (parameter space)
min_sigma = 1
max_sigma = 10
num_sigma = 30
threshold = 0.01

p = SpotFinder.BlobDetector(
    min_sigma=min_sigma,
    max_sigma=max_sigma,
    num_sigma=num_sigma,
    threshold=threshold,
    measurement_type='mean',
)

# detect triggers some numpy warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # blobs = dots; define the spots in the dots image, but then find them again in the stack.
    dots = dots.max_proj(Axes.ROUND, Axes.ZPLANE)
    dots_numpy = dots._squeezed_numpy(Axes.ROUND, Axes.ZPLANE)
    blobs_image = dots_numpy
    intensities = p.run(registered_image, blobs_image=blobs_image)
# EPY: END code

# EPY: START markdown
# The Encoder table is the hypothesized standardized file format for the output of a spot detector, and is the first output file format in the pipeline that is not an image or set of images
# EPY: END markdown

# EPY: START markdown
#`attributes` is produced by the encoder and contains all the information necessary to map the encoded spots back to the original image
#
#`x, y` describe the position, while `x_min` through `y_max` describe the bounding box for the spot, which is refined by a radius `r`. This table also stores the intensity and spot_id.
# EPY: END markdown

# EPY: START markdown
### Decode
# EPY: END markdown

# EPY: START markdown
#Each assay type also exposes a decoder. A decoder translates each spot (spot_id) in the Encoder table into a gene (that matches a barcode) and associates this information with the stored position. The goal is to decode and output a quality score that describes the confidence in the decoding.
# EPY: END markdown

# EPY: START markdown
#There are hard and soft decodings -- hard decoding is just looking for the max value in the code book. Soft decoding, by contrast, finds the closest code by distance (in intensity). Because different assays each have their own intensities and error modes, we leave decoders as user-defined functions.
# EPY: END markdown

# EPY: START code
decoded = experiment.codebook.decode_per_round_max(intensities)
# EPY: END code

# EPY: START markdown
### Compare to results from paper
# EPY: END markdown

# EPY: START markdown
#Besides house keeping genes, VIM and HER2 should be most highly expessed, which is consistent here.
# EPY: END markdown

# EPY: START code
genes, counts = np.unique(decoded.loc[decoded[Features.PASSES_THRESHOLDS]][Features.TARGET], return_counts=True)
table = pd.Series(counts, index=genes).sort_values(ascending=False)
# EPY: END code

# EPY: START markdown
#### Segment
# EPY: END markdown

# EPY: START markdown
#After calling spots and decoding their gene information, cells must be segmented to assign genes to cells. This paper used a seeded watershed approach.
# EPY: END markdown

# EPY: START code
from starfish.image import Segmentation

dapi_thresh = .16  # binary mask for cell (nuclear) locations
stain_thresh = .22  # binary mask for overall cells // binarization of stain
min_dist = 57

registered_mp = registered_image.max_proj(Axes.CH, Axes.ZPLANE)
registered_mp_numpy = registered_mp._squeezed_numpy(Axes.CH, Axes.ZPLANE)
stain = np.mean(registered_mp_numpy, axis=0)
stain = stain/stain.max()
nuclei = nuclei.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
nuclei_numpy = nuclei._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)

seg = Segmentation.Watershed(
    nuclei_threshold=dapi_thresh,
    input_threshold=stain_thresh,
    min_distance=min_dist
)
label_image = seg.run(registered_image, nuclei)
seg.show()
# EPY: END code

# EPY: START markdown
#### Assign spots to cells and create cell x gene count matrix
# EPY: END markdown

# EPY: START code
from starfish.spots import TargetAssignment
al = TargetAssignment.Label()
labeled = al.run(label_image, decoded)
# EPY: END code

# EPY: START code
from starfish.expression_matrix.expression_matrix import ExpressionMatrix
# EPY: END code

# EPY: START code
cg = labeled.to_expression_matrix()
cg
# EPY: END code

# EPY: START markdown
#Plot the (x, y) centroids of segmented cells in small cyan dots. Plot cells expressing VIM in blue, and cells expressing HER2 in red. Compare with the following plot of the displayed _spots_ below. This demonstrates that (1) the expression matrix is being properly created but (2) many of the spots are occuring outside segmented cells, suggesting that the segmentation may be too restrictive. 
# EPY: END markdown

# EPY: START code
vim_mask = cg.loc[:, 'VIM'] > 0
her2_mask = cg.loc[:, 'HER2'] > 0
plt.scatter(cg['x'], -cg['y'], s=5, c='c')
plt.scatter(cg['x'][vim_mask], -cg['y'][vim_mask], s=12, c='b')
plt.scatter(cg['x'][her2_mask], -cg['y'][her2_mask], s=12, c='r')

# EPY: END code

# EPY: START markdown
#### Visualize results
#
#This FOV was selected to make sure that we can visualize the tumor/stroma boundary, below this is described by pseudo-coloring `HER2` (tumor) and vimentin (`VIM`, stroma)
# EPY: END markdown

# EPY: START code
from skimage.color import rgb2gray

GENE1 = 'HER2'
GENE2 = 'VIM'

rgb = np.zeros(registered_image.tile_shape + (3,))
nuclei_mp = nuclei.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
nuclei_numpy = nuclei_mp._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
rgb[:,:,0] = nuclei_numpy
dots_mp = dots.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
dots_mp_numpy = dots_mp._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
rgb[:,:,1] = dots_mp_numpy
do = rgb2gray(rgb)
do = do/(do.max())

image(do,size=10)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    is_gene1 = decoded.where(decoded[Features.AXIS][Features.TARGET] == GENE1, drop=True)
    is_gene2 = decoded.where(decoded[Features.AXIS][Features.TARGET] == GENE2, drop=True)

plt.plot(is_gene1.x, is_gene1.y, 'or')
plt.plot(is_gene2.x, is_gene2.y, 'ob')
plt.title(f'Red: {GENE1}, Blue: {GENE2}');
# EPY: END code
