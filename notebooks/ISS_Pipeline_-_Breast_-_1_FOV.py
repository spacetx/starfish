#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"hide_input": false, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}, "toc": {"nav_menu": {}, "number_sections": true, "sideBar": true, "skip_h1_title": false, "toc_cell": false, "toc_position": {}, "toc_section_display": "block", "toc_window_display": false}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# ## Reproduce In-situ Sequencing results with Starfish
# 
# This notebook walks through a work flow that reproduces an ISS result for one field of view using the starfish package.
# 
# ## Load tiff stack and visualize one field of view
# EPY: END markdown

# EPY: START code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from showit import image, tile
import pprint
# EPY: ESCAPE %matplotlib inline

from starfish.io import Stack
from starfish.constants import Indices
from starfish.codebook import Codebook
# EPY: END code

# EPY: START code
s = Stack()
s.read('https://dmf0bdeheu4zf.cloudfront.net/20180716/ISS/fov_001/experiment.json')
# s.image.squeeze() simply converts the 4D tensor H*C*X*Y into a list of len(H*C) image planes for rendering by 'tile'
tile(s.image.squeeze());
# EPY: END code

# EPY: START markdown
# ## Show input file format that specifies how the tiff stack is organized
# 
# The stack contains multiple single plane images, one for each color channel, 'ch', (columns in above image) and hybridization round, 'hyb', (rows in above image). This protocol assumes that genes are encoded with a length 4 quatenary barcode that can be read out from the images. Each hybridization encodes a position in the codeword. The maximum signal in each color channel (columns in the above image) corresponds to a letter in the codeword. The channels, in order, correspond to the letters: 'T', 'G', 'C', 'A'. The goal is now to process these image data into spatially organized barcodes, e.g., ACTG, which can then be mapped back to a codebook that specifies what gene this codeword corresponds to.
# EPY: END markdown

# EPY: START code
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(s.org)
# EPY: END code

# EPY: START markdown
# The flat TIFF files are loaded into a 4-d tensor with dimensions corresponding to hybridization round, channel, x, and y. For other volumetric approaches that image the z-plane, this would be a 5-d tensor.
# EPY: END markdown

# EPY: START code
# hyb, channel, x, y, z
s.image.numpy_array.shape
# EPY: END code

# EPY: START markdown
# ## Show auxiliary images captured during the experiment
# EPY: END markdown

# EPY: START markdown
# 'dots' is a general stain for all possible transcripts. This image should correspond to the maximum projcection of all color channels within a single hybridization round. This auxiliary image is useful for registering images from multiple hybridization rounds to this reference image. We'll see an example of this further on in the notebook
# EPY: END markdown

# EPY: START code
image(s.auxiliary_images['dots'].max_proj(Indices.ROUND, Indices.CH, Indices.Z))
# EPY: END code

# EPY: START markdown
# Below is a DAPI auxiliary image, which specifically marks nuclei. This is useful cell segmentation later on in the processing.
# EPY: END markdown

# EPY: START code
image(s.auxiliary_images['nuclei'].max_proj(Indices.ROUND, Indices.CH, Indices.Z))
# EPY: END code

# EPY: START markdown
# ## Examine the codebook
# EPY: END markdown

# EPY: START markdown
# Each 4 letter quatenary code (as read out from the 4 hybridization rounds and 4 color channels) represents a gene. This relationship is stored in a codebook
# EPY: END markdown

# EPY: START code
codebook = Codebook.from_json('https://s3.amazonaws.com/czi.starfish.data.public/20180716/ISS/codebook.json')
codebook
# EPY: END code

# EPY: START markdown
# ## Filter and scale raw data
# 
# Now apply the white top hat filter to both the spots image and the individual channels. White top had enhances white spots on a black background.
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter import Filter

# filter raw data
disk_size = 15  # disk as in circle
filt = Filter.WhiteTophat(disk_size, verbose=True)
filt.filter(s.image)
for img in s.auxiliary_images.values():
    filt.filter(img)
# EPY: END code

# EPY: START markdown
# ## Register data
# EPY: END markdown

# EPY: START markdown
# For each hybridization round, the max projection across color channels should look like the dots stain.
# Below, this computes the max projection across the color channels of a hybridization round and learns the linear transformation to maps the resulting image onto the dots image.
# 
# The Fourier shift registration approach can be thought of as maximizing the cross-correlation of two images.
# 
# In the below table, Error is the minimum mean-squared error, and shift reports changes in x and y dimension.
# EPY: END markdown

# EPY: START code
from starfish.pipeline.registration import Registration

registration = Registration.FourierShiftRegistration(upsampling=1000, reference_stack=s.auxiliary_images['dots'], verbose=True)
registration.register(s.image)
# EPY: END code

# EPY: START markdown
# ## Use spot-detector to create 'encoder' table  for standardized input  to decoder
# EPY: END markdown

# EPY: START markdown
# Each pipeline exposes an encoder that translates an image into spots with intensities.  This approach uses a Gaussian spot detector.
# EPY: END markdown

# EPY: START code
from starfish.pipeline.features.spots.detector import SpotFinder
import warnings

# parameters to define the allowable gaussian sizes (parameter space)
min_sigma = 1
max_sigma = 10
num_sigma = 30
threshold = 0.01

p = SpotFinder.GaussianSpotDetector(
    min_sigma=min_sigma,
    max_sigma=max_sigma,
    num_sigma=num_sigma,
    threshold=threshold,
    blobs_stack=s.auxiliary_images['dots'],
    measurement_type='mean',
)

# detect triggers some numpy warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # blobs = dots; define the spots in the dots image, but then find them again in the stack.
    intensities = p.find(s.image)
# EPY: END code

# EPY: START code
# Verify the spot count is reasonable.
spot_count = intensities.shape[0]
assert 1000 < spot_count < 5000
spot_count
# EPY: END code

# EPY: START markdown
# This visualizes a single spot (#100) across all hybridization rounds and channels. It contains the intensity and bit index, which allow it to be mapped onto the correct barcode.
# EPY: END markdown

# EPY: START code
intensities[100]
# EPY: END code

# EPY: START markdown
# The Encoder table is the hypothesized standardized file format for the output of a spot detector, and is the first output file format in the pipeline that is not an image or set of images
# EPY: END markdown

# EPY: START markdown
# `attributes` is produced by the encoder and contains all the information necessary to map the encoded spots back to the original image
# 
# `x, y` describe the position, while `x_min` through `y_max` describe the bounding box for the spot, which is refined by a radius `r`. This table also stores the intensity and spot_id.
# EPY: END markdown

# EPY: START markdown
# ## Decode
# EPY: END markdown

# EPY: START markdown
# Each assay type also exposes a decoder. A decoder translates each spot (spot_id) in the Encoder table into a gene (that matches a barcode) and associates this information with the stored position. The goal is to decode and output a quality score that describes the confidence in the decoding.
# EPY: END markdown

# EPY: START markdown
# There are hard and soft decodings -- hard decoding is just looking for the max value in the code book. Soft decoding, by contrast, finds the closest code by distance (in intensity). Because different assays each have their own intensities and error modes, we leave decoders as user-defined functions.
# EPY: END markdown

# EPY: START code
decoded = codebook.decode_per_hyb_max(intensities)
# EPY: END code

# EPY: START markdown
# ## Compare to results from paper
# EPY: END markdown

# EPY: START markdown
# Besides house keeping genes, VIM and HER2 should be most highly expessed, which is consistent here.
# EPY: END markdown

# EPY: START code
genes, counts = np.unique(decoded['gene_name'], return_counts=True)
table = pd.Series(counts, index=genes).sort_values(ascending=False)
# EPY: END code

# EPY: START code
table.head()
# EPY: END code

# EPY: START code
assert table.index.get_loc('HER2') < 10
assert table.index.get_loc('VIM') < 10
table.head()
# EPY: END code

# EPY: START markdown
# ### Segment
# EPY: END markdown

# EPY: START markdown
# After calling spots and decoding their gene information, cells must be segmented to assign genes to cells. This paper used a seeded watershed approach.
# EPY: END markdown

# EPY: START code
from starfish.constants import Indices
from starfish.pipeline.segmentation.watershed import _WatershedSegmenter

dapi_thresh = .16  # binary mask for cell (nuclear) locations
stain_thresh = .22  # binary mask for overall cells // binarization of stain
size_lim = (10, 10000)
disk_size_markers = None
disk_size_mask = None
min_dist = 57

stain = np.mean(s.image.max_proj(Indices.CH, Indices.Z), axis=0)
stain = stain/stain.max()
nuclei = s.auxiliary_images['nuclei'].max_proj(Indices.ROUND, Indices.CH, Indices.Z)


seg = _WatershedSegmenter(nuclei, stain)  # uses skimage watershed.
cells_labels = seg.segment(dapi_thresh, stain_thresh, size_lim, disk_size_markers, disk_size_mask, min_dist)
seg.show()
# EPY: END code

# EPY: START markdown
# ### Visualize results
# 
# This FOV was selected to make sure that we can visualize the tumor/stroma boundary, below this is described by pseudo-coloring `HER2` (tumor) and vimentin (`VIM`, stroma)
# EPY: END markdown

# EPY: START code
from skimage.color import rgb2gray

GENE1 = 'HER2'
GENE2 = 'VIM'

rgb = np.zeros(s.image.tile_shape + (3,))
rgb[:,:,0] = s.auxiliary_images['nuclei'].max_proj(Indices.ROUND, Indices.CH, Indices.Z)
rgb[:,:,1] = s.auxiliary_images['dots'].max_proj(Indices.ROUND, Indices.CH, Indices.Z)
do = rgb2gray(rgb)
do = do/(do.max())

image(do,size=10)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    is_gene1 = decoded.where(decoded.gene_name == GENE1, drop=True)
    is_gene2 = decoded.where(decoded.gene_name == GENE2, drop=True)

plt.plot(is_gene1.x, is_gene1.y, 'or')
plt.plot(is_gene2.x, is_gene2.y, 'ob')
plt.title(f'Red: {GENE1}, Blue: {GENE2}');
# EPY: END code
