#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Reproduce DARTFISH results with Starfish
#
#DARTFISH is a multiplexed image based transcriptomics assay from the [Zhang lab](http://genome-tech.ucsd.edu/ZhangLab/). As of this writing, this assay is not published yet. Nevertheless, here we demonstrate that Starfish can be used to process the data from raw images into spatially resolved gene expression profiles
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: ESCAPE %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from starfish import display
from starfish import data, FieldOfView
from starfish.types import Axes, Features, Levels

from starfish import IntensityTable

from starfish.image import Filter
from starfish.spots import DetectPixels
# EPY: END code

# EPY: START markdown
### Load data into Starfish from the Cloud
#
#The example data here corresopond to DARTFISHv1 2017. The group is actively working on improving the protocol. The data represent human brain tissue from the human occipital cortex from 1 field of view (FOV) of larger experiment. The data from one field of view correspond to 18 images from 6 imaging rounds (r) 3 color channels (c) and 1 z-plane (z). Each image is 988x988 (y,x)
# EPY: END markdown

# EPY: START code
use_test_data = os.getenv("USE_TEST_DATA") is not None
experiment = data.DARTFISH(use_test_data=use_test_data)

imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

print(imgs)
# EPY: END code

# EPY: START markdown
### Visualize codebook
#
#The DARTFISH codebook maps pixel intensities across the rounds and channels to the corresponding barcodes and genes that those pixels code for. For this example dataset, the codebook specifies 96 possible barcodes. The codebook used in this experiment has 3 color channels and one blank channel, each of which contribute to codes. The presence of the blank channel will be important later when the filtering is described.
# EPY: END markdown

# EPY: START code
experiment.codebook
# EPY: END code

# EPY: START markdown
### Visualize raw data
#
#A nice way to page through all this data is to use the display command. We have commented this out for now, because it will not render in Github. Instead, we simply show an image from the first round and color channel.
# EPY: END markdown

# EPY: START code
# %gui qt5
# display(stack)

single_plane = imgs.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 0})
single_plane = single_plane.xarray.squeeze()
plt.figure(figsize=(7,7))
plt.imshow(single_plane, cmap='gray', clim=list(np.percentile(single_plane, [1, 99.9])))
plt.title('Round: 0, Chanel:0')
plt.axis('off');
# EPY: END code

# EPY: START markdown
### Filter and scale raw data before decoding into spatially resolved gene expression
#
#First, we equalize the intensity of the images by scaling each image by its maximum intensity, which is equivalent to scaling by the 100th percentile value of the pixel values in each image.
# EPY: END markdown

# EPY: START code
sc_filt = Filter.Clip(p_max=100, level_method=Levels.SCALE_BY_CHUNK)
norm_imgs = sc_filt.run(imgs)
# EPY: END code

# EPY: START markdown
#Next, for each imaging round, and each pixel location, we zero the intensity values across all three color channels if the magnitude of this 3 vector is below a threshold. As such, the code value associated with these pixels will be the blank. This is necessary to support euclidean decoding for codebooks that include blank values.
# EPY: END markdown

# EPY: START code
z_filt = Filter.ZeroByChannelMagnitude(thresh=.05, normalize=False)
filtered_imgs = z_filt.run(norm_imgs)
# EPY: END code

# EPY: START markdown
### Decode the processed data into spatially resolved gene expression profiles
#
#Here, starfish decodes each pixel value, across all rounds and channels, into the corresponding target (gene) it corresponds too. Contiguous pixels that map to the same target gene are called as one RNA molecule. Intuitively, pixel vectors are matched to the codebook by computing the euclidean distance between the pixel vector and all codewords. The minimal distance gene target is then selected, if it is within `distance_threshold` of any code.
#
#This decoding operation requires some parameter tuning, which is described below. First, we look at a distribution of pixel vector barcode magnitudes to determine the minimum magnitude threshold at which we will attempt to decode the pixel vector.
# EPY: END markdown

# EPY: START code
def compute_magnitudes(stack, norm_order=2):

    pixel_intensities = IntensityTable.from_image_stack(stack)
    feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
    norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

    return norm

mags = compute_magnitudes(filtered_imgs)

plt.hist(mags, bins=20);
sns.despine(offset=3)
plt.xlabel('Barcode magnitude')
plt.ylabel('Number of pixels')
plt.yscale('log');
# EPY: END code

# EPY: START markdown
#Next, we decode the the data
# EPY: END markdown

# EPY: START code
# how much magnitude should a barcode have for it to be considered by decoding? this was set by looking at
# the plot above
magnitude_threshold = 0.5
# how big do we expect our spots to me, min/max size. this was set to be equivalent to the parameters
# determined by the Zhang lab.
area_threshold = (5, 30)
# how close, in euclidean space, should the pixel barcode be to the nearest barcode it was called to?
# here, I set this to be a large number, so I can inspect the distribution of decoded distances below
distance_threshold = 3

psd = DetectPixels.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

initial_spot_intensities, results = psd.run(filtered_imgs)
# EPY: END code

# EPY: START code
spots_df = initial_spot_intensities.to_features_dataframe()
spots_df['area'] = np.pi*spots_df['radius']**2
spots_df = spots_df.loc[spots_df[Features.PASSES_THRESHOLDS]]
spots_df.head()
# EPY: END code

# EPY: START markdown
### Compare to benchmark results
#
#The below plot aggregates gene copy number across cells in the field of view and compares the results to the same copy numbers from the authors' pipeline. This can likely be improved by tweaking parameters in the above algorithms.
# EPY: END markdown

# EPY: START code
# load results from authors' pipeline
# instead of just calling read_csv with the url, we are using python requests to load it to avoid a
# SSL certificate error on some platforms.
import io, requests
cnts_benchmark = pd.read_csv(io.BytesIO(requests.get('https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH/fov_001/counts.csv').content))
cnts_benchmark.head()

# select spots with distance less than a threshold, and count the number of each target gene
min_dist = 0.6
cnts_starfish = spots_df[spots_df.distance<=min_dist].groupby('target').count()['area']
cnts_starfish = cnts_starfish.reset_index(level=0)
cnts_starfish.rename(columns = {'target':'gene', 'area':'cnt_starfish'}, inplace=True)

benchmark_comparison = pd.merge(cnts_benchmark, cnts_starfish, on='gene', how='left')
benchmark_comparison.head(20)

x = benchmark_comparison.dropna().cnt.values
y = benchmark_comparison.dropna().cnt_starfish.values
r = np.corrcoef(x, y)
r = r[0,1]

plt.scatter(x, y, 50,zorder=2)

plt.xlabel('Gene copy number Benchmark')
plt.ylabel('Gene copy number Starfish')
plt.title('r = {}'.format(r))

sns.despine(offset=2)
# EPY: END code

# EPY: START markdown
### Visualize results
#
#This image applies a pseudo-color to each gene channel to visualize the position and size of all called spots in the test image
# EPY: END markdown

# EPY: START code
# exclude spots that don't meet our area thresholds
area_lookup = lambda x: 0 if x == 0 else results.region_properties[x - 1].area
vfunc = np.vectorize(area_lookup)
mask = np.squeeze(vfunc(results.label_image))
new_image = np.squeeze(results.decoded_image)*(mask > area_threshold[0])*(mask < area_threshold[1])

plt.figure(figsize=(10,10))
plt.imshow(new_image, cmap = 'nipy_spectral');
plt.axis('off');
plt.title('Coded rolonies');

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

rect = [Rectangle((100, 600), width=200, height=200)]
pc = PatchCollection(rect, facecolor='none', alpha=1.0, edgecolor='w', linewidth=1.5)
plt.gca().add_collection(pc)

plt.figure(figsize=(10,10))
plt.imshow(new_image[600:800, 100:300], cmap = 'nipy_spectral');
plt.axis('off');
plt.title('Coded rolonies, zoomed in');
# EPY: END code

# EPY: START markdown
### Parameter and QC analysis
#
#Here, we further investigate reasonable choices for each of the parameters used by the PixelSpotDecoder. By tuning these parameters, one can achieve different results
# EPY: END markdown

# EPY: START code
plt.figure(figsize=(10,3))

plt.subplot(131)
plt.hist(mags, bins=100);
plt.yscale('log')
plt.xlabel('barcode magnitude')
plt.ylabel('number of pixels')
sns.despine(offset=2)
plt.vlines(magnitude_threshold, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1])
plt.title('Set magnitude threshod')

plt.subplot(132)
spots_df['area'] = np.pi*spots_df.radius**2
spots_df.area.hist(bins=30);
plt.xlabel('area')
plt.ylabel('number of spots')
sns.despine(offset=2)
plt.title('Set area threshold')

plt.subplot(133)
spots_df.distance.hist(bins=30)
plt.xlabel('min distance to code');
plt.vlines(min_dist, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1])
sns.despine(offset=2)
plt.title('Set minimum distance threshold');
# EPY: END code

# EPY: START code
distance_threshold = min_dist

psd = DetectPixels.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

spot_intensities, results = psd.run(filtered_imgs)
spot_intensities = IntensityTable(spot_intensities.where(spot_intensities[Features.PASSES_THRESHOLDS], drop=True))
# EPY: END code

# EPY: START markdown
#Here, we:
#
#1. Pick a rolony that was succesfully decoded to a gene.
#2. Pull out the average pixel trace for that rolony
#3. Plot that pixel trace against the barcode of that gene
#
#In order to assess, visually, how close decoded barcodes match their targets
# EPY: END markdown

# EPY: START code
# reshape the spot intensity table into a RxC barcode vector
pixel_traces = spot_intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))

# extract dataframe from spot intensity table for indexing purposes
pixel_traces_df = pixel_traces.to_features_dataframe()
pixel_traces_df['area'] = np.pi*pixel_traces_df.radius**2

# pick index of a barcode that was read and decoded from the ImageStack
ind = 4

# get the the corresponding gene this barcode was decoded to
gene = pixel_traces_df.loc[ind].target

# query the codebook for the actual barcode corresponding to this gene
real_barcode = experiment.codebook[experiment.codebook.target==gene].stack(traces=(Axes.ROUND.value, Axes.CH.value)).values[0]
read_out_barcode = pixel_traces[ind,:]

plt.plot(real_barcode, 'ok')
plt.stem(read_out_barcode)
sns.despine(offset=2)
plt.xticks(range(18))
plt.title(gene)
plt.xlabel('Index into R (0:5) and C(0:2)');
# EPY: END code
