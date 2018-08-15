#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# ## Reproduce DARTFISH results with a Pixel Decoding Method
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: ESCAPE %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from starfish.experiment import Experiment
from starfish.codebook import Codebook
from starfish.constants import Indices

from starfish.intensity_table import IntensityTable

from starfish.stack import ImageStack
from starfish.image import Filter
from starfish.spots import SpotFinder
# EPY: END code

# EPY: START markdown
# ### Load image stack
# 
# Note that the data here corresopond to DARTFISHv1 2017. The group is actively working on improving the protocol.
# EPY: END markdown

# EPY: START code
exp = Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180813/DARTFISH/fov_001/experiment.json')
stack = exp.image
# TODO the latter will be fixed by https://github.com/spacetx/starfish/issues/316
stack._data = stack._data.astype(float)
# EPY: END code

# EPY: START code
print(stack.shape)
# EPY: END code

# EPY: START code
stack.show_stack({Indices.CH:0}, rescale=True)
# EPY: END code

# EPY: START markdown
# ### Load codebook
# EPY: END markdown

# EPY: START code
cb = Codebook.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180813/DARTFISH/fov_001/codebook.json')
cb
# EPY: END code

# EPY: START markdown
# ### Load copy number benchmark results
# EPY: END markdown

# EPY: START code
bench = pd.read_csv('https://dmf0bdeheu4zf.cloudfront.net/20180813/DARTFISH/fov_001/counts.csv')
bench.head()
# EPY: END code

# EPY: START markdown
# ### Filter Image Stack
# EPY: END markdown

# EPY: START code
sc_filt = Filter.ScaleByPercentile(p=100)
z_filt = Filter.ZeroByChannelMagnitude(thresh=.05, normalize=False)

norm_stack = sc_filt.run(stack, in_place=False)
zero_norm_stack = z_filt.run(norm_stack, in_place=False)
# EPY: END code

# EPY: START markdown
# #### Visualize barcode magnitudes to help determine an appropriate threshold for decoding
# EPY: END markdown

# EPY: START code
def compute_magnitudes(stack, norm_order=2):
    
    pixel_intensities = IntensityTable.from_image_stack(zero_norm_stack)
    feature_traces = pixel_intensities.stack(traces=(Indices.CH.value, Indices.ROUND.value))
    norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

    return norm

mags = compute_magnitudes(zero_norm_stack)

plt.hist(mags, bins=20);
sns.despine(offset=3)
plt.xlabel('Barcode magnitude')
plt.ylabel('Number of pixels');
plt.yscale('log')
# EPY: END code

# EPY: START markdown
# ### Decode
# EPY: END markdown

# EPY: START code
magnitude_threshold = 0.5
area_threshold = (5, 30)
distance_threshold = 3

psd = SpotFinder.PixelSpotDetector(
    codebook=cb,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

spot_intensities, results = psd.find(zero_norm_stack)
spots_df = spot_intensities.to_dataframe()
spots_df['area'] = np.pi*spots_df['radius']**2
spots_df.head()
# EPY: END code

# EPY: START markdown
# ### QC Plots
# EPY: END markdown

# EPY: START markdown
# #### parameter tuning plots
# EPY: END markdown

# EPY: START code
min_dist = 0.6

plt.figure(figsize=(10,3))

plt.subplot(131)
plt.hist(mags, bins=100);
plt.yscale('log')
plt.xlabel('barcode magnitude')
plt.ylabel('number of pixels')
sns.despine(offset=2)
plt.vlines(magnitude_threshold, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1])

plt.subplot(132)
spots_df['area'] = np.pi*spots_df.radius**2
spots_df.area.hist(bins=30);
plt.xlabel('area')
plt.ylabel('number of spots')
sns.despine(offset=2)

plt.subplot(133)
spots_df.distance.hist(bins=30)
plt.xlabel('min distance to code');
plt.vlines(min_dist, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1])
sns.despine(offset=2)
# EPY: END code

# EPY: START markdown
# #### Copy number comparisons
# EPY: END markdown

# EPY: START code
cnts_starfish = spots_df[spots_df.distance<=min_dist].groupby('target').count()['area']
cnts_starfish = cnts_starfish.reset_index(level=0)
cnts_starfish.rename(columns = {'target':'gene', 'area':'cnt_starfish'}, inplace=True)

mrg = pd.merge(bench, cnts_starfish, on='gene', how='left')
mrg.head(20)
# EPY: END code

# EPY: START code
sns.set_context('talk')
sns.set_style('ticks')

x = mrg.dropna().cnt.values
y = mrg.dropna().cnt_starfish.values
r = np.corrcoef(x, y)
r = r[0,1]

plt.scatter(x, y, 50,zorder=2)

plt.xlabel('Gene copy number Benchmark')
plt.ylabel('Gene copy number Starfish')
plt.title('r = {}'.format(r))

sns.despine(offset=2)
# EPY: END code

# EPY: START markdown
# #### visualization of rolonies
# EPY: END markdown

# EPY: START code
distance_threshold = min_dist

psd = SpotFinder.PixelSpotDetector(
    codebook=cb,
    metric='euclidean',
    distance_threshold=distance_threshold,
    magnitude_threshold=magnitude_threshold,
    min_area=area_threshold[0],
    max_area=area_threshold[1]
)

spot_intensities, results = psd.find(zero_norm_stack)
# EPY: END code

# EPY: START code
area_lookup = lambda x: 0 if x == 0 else results.region_properties[x - 1].area
vfunc = np.vectorize(area_lookup)
mask = np.squeeze(vfunc(results.label_image))
new_image = np.squeeze(results.decoded_image)*(mask > area_threshold[0])*(mask < area_threshold[1])

plt.figure(figsize=(10,10))
plt.imshow(new_image, cmap = 'nipy_spectral');
plt.axis('off');
plt.title('Coded rolonies');

plt.figure(figsize=(10,10))
plt.imshow(new_image[600:800, 100:300], cmap = 'nipy_spectral');
plt.axis('off');
plt.title('Coded rolonies, zoomed in');
# EPY: END code

# EPY: START markdown
# ### visualization of matched barcodes
# EPY: END markdown

# EPY: START code
# reshape the spot intensity table into a RxC barcode vector
pbcs = spot_intensities.stack(traces=(Indices.ROUND.value, Indices.CH.value))

# extract dataframe from spot intensity table for indexing purposes
pbcs_df = pbcs.to_dataframe()
pbcs_df['area'] = np.pi*pbcs_df.radius**2

# pick index of a random barcode that was read and decoded from the ImageStack
ind = int(np.ceil(np.random.rand()*len(pbcs_df)))-1

# get the the corresponding gene this barcode was decoded to
gene = pbcs_df.loc[ind].target

# query the codebook for the actual barcode corresponding to this gene
real_barcode = cb[cb.target==gene].stack(traces=(Indices.ROUND.value, Indices.CH.value)).values[0]
read_out_barcode = pbcs[ind,:]

plt.plot(real_barcode, 'ok')
plt.stem(read_out_barcode)
plt.title(gene);
# EPY: END code
