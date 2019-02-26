#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Reproduce DARTFISH results with a Pixel Decoding Method
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

import starfish.display
from starfish import data, FieldOfView
from starfish.types import Features, Axes

from starfish import IntensityTable

from starfish.image import Filter
from starfish.spots import PixelSpotDecoder

sns.set_context('talk')
sns.set_style('ticks')
# EPY: END code

# EPY: START markdown
#### Load image stack
#
#Note that the data here corresopond to DARTFISHv1 2017. The group is actively working on improving the protocol.
# EPY: END markdown

# EPY: START code
use_test_data = os.getenv("USE_TEST_DATA") is not None
exp = data.DARTFISH(use_test_data=use_test_data)

stack = exp.fov()[FieldOfView.PRIMARY_IMAGES]
# EPY: END code

# EPY: START code
print(stack.shape)
# EPY: END code

# EPY: START code
starfish.display.stack(stack)
# EPY: END code

# EPY: START markdown
#### Load codebook
# EPY: END markdown

# EPY: START code
exp.codebook
# EPY: END code

# EPY: START markdown
#### Load copy number benchmark results
# EPY: END markdown

# EPY: START code
cnts_benchmark = pd.read_csv('https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH/fov_001/counts.csv')
cnts_benchmark.head()
# EPY: END code

# EPY: START markdown
#### Filter Image Stack
# EPY: END markdown

# EPY: START code
sc_filt = Filter.ScaleByPercentile(p=100)
z_filt = Filter.ZeroByChannelMagnitude(thresh=.05, normalize=False)

norm_stack = sc_filt.run(stack)
zero_norm_stack = z_filt.run(norm_stack)
# EPY: END code

# EPY: START markdown
##### Visualize barcode magnitudes to help determine an appropriate threshold for decoding
# EPY: END markdown

# EPY: START code
def compute_magnitudes(stack, norm_order=2):

    pixel_intensities = IntensityTable.from_image_stack(zero_norm_stack)
    feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
    norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

    return norm

mags = compute_magnitudes(zero_norm_stack)

plt.hist(mags, bins=20);
sns.despine(offset=3)
plt.xlabel('Barcode magnitude')
plt.ylabel('Number of pixels')
plt.yscale('log');
# EPY: END code

# EPY: START markdown
#### Decode
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

psd = PixelSpotDecoder.PixelSpotDecoder(
    codebook=exp.codebook,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

initial_spot_intensities, results = psd.run(zero_norm_stack)
# EPY: END code

# EPY: START code
spots_df = initial_spot_intensities.to_features_dataframe()
spots_df['area'] = np.pi*spots_df['radius']**2
spots_df = spots_df.loc[spots_df[Features.PASSES_THRESHOLDS]]
spots_df.head()
# EPY: END code

# EPY: START markdown
#### QC Plots
# EPY: END markdown

# EPY: START markdown
##### parameter tuning plots
# EPY: END markdown

# EPY: START code
# these plots help inform how the parameters above were wet.
# looking at the last plot below, I reset the distance_threshold parameter to
min_dist = 0.6

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

# EPY: START markdown
##### Copy number comparisons
# EPY: END markdown

# EPY: START code
# select spots with distance less than a threshold, and count the number of each target gene
cnts_starfish = spots_df[spots_df.distance<=min_dist].groupby('target').count()['area']
cnts_starfish = cnts_starfish.reset_index(level=0)
cnts_starfish.rename(columns = {'target':'gene', 'area':'cnt_starfish'}, inplace=True)

benchmark_comparison = pd.merge(cnts_benchmark, cnts_starfish, on='gene', how='left')
benchmark_comparison.head(20)
# EPY: END code

# EPY: START code
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
##### visualization of rolonies
# EPY: END markdown

# EPY: START code
distance_threshold = min_dist

psd = PixelSpotDecoder.PixelSpotDecoder(
    codebook=exp.codebook,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

spot_intensities, results = psd.run(zero_norm_stack)
spot_intensities = IntensityTable(spot_intensities.where(spot_intensities[Features.PASSES_THRESHOLDS], drop=True))
# EPY: END code

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
#### visualization of matched barcodes
#here, we 1. pick a rolony that was succesfully decoded to a gene. 2. pull out the average pixel trace for that rolony and 3. plot that pixel trace against the barcode of that gene
# EPY: END markdown

# EPY: START code
# reshape the spot intensity table into a RxC barcode vector
pixel_traces = spot_intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))

# extract dataframe from spot intensity table for indexing purposes
pixel_traces_df = pixel_traces.to_features_dataframe()
pixel_traces_df['area'] = np.pi*pixel_traces_df.radius**2

# pick index of a barcode that was read and decoded from the ImageStack
ind = 4

# The test will error here on pixel_traces[ind,:] with an out of index error
# because we are using the test data.

# get the the corresponding gene this barcode was decoded to
gene = pixel_traces_df.loc[ind].target

# query the codebook for the actual barcode corresponding to this gene
real_barcode = exp.codebook[exp.codebook.target==gene].stack(traces=(Axes.ROUND.value, Axes.CH.value)).values[0]
read_out_barcode = pixel_traces[ind,:]

plt.plot(real_barcode, 'ok')
plt.stem(read_out_barcode)
sns.despine(offset=2)
plt.xticks(range(18))
plt.title(gene)
plt.xlabel('Index into R (0:5) and C(0:2)');
# EPY: END code
