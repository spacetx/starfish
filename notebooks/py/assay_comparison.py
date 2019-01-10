#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
# plotting
# EPY: ESCAPE %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')
# EPY: END code

# EPY: START code
# munging
import os
import requests
import tempfile
# EPY: END code

# EPY: START code
# science
from starfish import IntensityTable, Experiment, FieldOfView, ImageStack
from starfish.plot import histogram, compare_copy_number
from starfish.plot.decoded_spots import decoded_spots
from starfish.types import Features, Axes
import numpy as np
import pandas as pd

from showit import image
# EPY: END code

# EPY: START markdown
### Load IntensityTables
# EPY: END markdown

# EPY: START code
# IntensityTable can't download from directories without list privileges

data_root = "https://d2nhj9g34unfro.cloudfront.net/assay_comparison/"
iss_link = os.path.join(data_root, "iss.nc")
merfish_link = os.path.join(data_root, "merfish.nc")
dartfish_link = os.path.join(data_root, "dartfish.nc")

tmp = tempfile.gettempdir()
iss_nc = os.path.join(tmp, "iss.nc")
merfish_nc = os.path.join(tmp, "merfish.nc")
dartfish_nc = os.path.join(tmp, "dartfish.nc")


def curl(dest_path, link):
    with open(dest_path, "wb") as fh:
        fh.write(requests.get(link).content)


curl(iss_nc, iss_link)
curl(merfish_nc, merfish_link)
curl(dartfish_nc, dartfish_link)

iss_intensity_table = IntensityTable.load(iss_nc)
merfish_intensity_table = IntensityTable.load(merfish_nc)
dartfish_intensity_table = IntensityTable.load(dartfish_nc)
# EPY: END code

# EPY: START code
datasets = [iss_intensity_table, merfish_intensity_table, dartfish_intensity_table]
# EPY: END code

# EPY: START markdown
### Load Background Images
# EPY: END markdown

# EPY: START code
# construct background images for each assay
import starfish.data
experiment = starfish.data.DARTFISH()

dartfish_nuclei_mp = experiment.fov()['nuclei'].max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
dartfish_nuclei_mp_numpy = dartfish_nuclei_mp._squeezed_numpy(Axes.CH, Axes.ROUND, Axes.ZPLANE)
dartfish_link = os.path.join(data_root, "dartfish_dots_image.npy")
dartfish_npy = os.path.join(tmp, "dartfish.npy")
curl(dartfish_npy, dartfish_link)
dartfish_dots = np.load(dartfish_npy)
# EPY: END code

# EPY: START code
experiment = starfish.data.ISS()

iss_nuclei_mp = experiment.fov()['nuclei'].max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
iss_nuclei_mp_numpy = iss_nuclei_mp._squeezed_numpy(Axes.CH, Axes.ROUND, Axes.ZPLANE)
iss_dots_mp = experiment.fov()['dots'].max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
iss_dots_mp_numpy = iss_dots_mp._squeezed_numpy(Axes.CH, Axes.ROUND, Axes.ZPLANE)
# EPY: END code

# EPY: START code
experiment = starfish.data.MERFISH()
merfish_nuclei_mp = experiment.fov()['nuclei'].max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
merfish_nuclei__mp_numpy = merfish_nuclei_mp._squeezed_numpy(Axes.CH, Axes.ROUND, Axes.ZPLANE)
# merfish doesn't have a dots image, and some of the channels are stronger than others.
# We can use the scale factors to get the right levels
merfish_background = experiment.fov()[FieldOfView.PRIMARY_IMAGES].max_proj(Axes.CH, Axes.ROUND)


from starfish.image import Filter
clip = Filter.Clip(p_max=99.7)
merfish_dots = clip.run(merfish_background)

merfish_mp = merfish_dots.max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
merfish_mp_numpy = merfish_mp._squeezed_numpy(Axes.CH, Axes.ROUND, Axes.ZPLANE)
# EPY: END code

# EPY: START markdown
### Load Decoded Images
# EPY: END markdown

# EPY: START markdown
#Numpy load can't download files from s3 either.
# EPY: END markdown

# EPY: START code
merfish_link = os.path.join(data_root, "merfish_decoded_image.npy")
dartfish_link = os.path.join(data_root, "dartfish_decoded_image.npy")

merfish_npy = os.path.join(tmp, "merfish_decoded_image.npy")
dartfish_npy = os.path.join(tmp, "dartfish_decoded_image.npy")


curl(merfish_npy, merfish_link)
curl(dartfish_npy, dartfish_link)


merfish_decoded_image = np.squeeze(np.load(merfish_npy))
dartfish_decoded_image = np.squeeze(np.load(dartfish_npy))
# EPY: END code

# EPY: START markdown
### Show Different Background Types for MERFISH
# EPY: END markdown

# EPY: START code
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, figsize=(30, 45))
decoded_spots(
    merfish_intensity_table,
    background_image=np.zeros_like(merfish_mp_numpy),
    spots_kwargs=dict(alpha=1.),
    ax=ax1
)
decoded_spots(
    merfish_intensity_table,
    background_image=merfish_mp_numpy,
    spots_kwargs=dict(alpha=1.),
    ax=ax3
)
decoded_spots(
    merfish_intensity_table,
    background_image=merfish_nuclei__mp_numpy,
    spots_kwargs=dict(alpha=1.),
    ax=ax5
)
decoded_spots(
    decoded_image=merfish_decoded_image,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax2
)
decoded_spots(
    decoded_image=merfish_decoded_image,
    background_image=merfish_mp_numpy,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax4
)
decoded_spots(
    decoded_image=merfish_decoded_image,
    background_image=merfish_nuclei__mp_numpy,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax6
);
# EPY: END code

# EPY: START markdown
#From these examples, we can see that the point cloud over-estimates the spot size
#(perhaps we're calculating radius wrong?)
# EPY: END markdown

# EPY: START markdown
### Show different background types for DARTFISH
# EPY: END markdown

# EPY: START code
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, figsize=(30, 45))
decoded_spots(
    dartfish_intensity_table,
    background_image=np.zeros_like(dartfish_dots),
    spots_kwargs=dict(alpha=1.),
    ax=ax1
)
decoded_spots(
    dartfish_intensity_table,
    background_image=dartfish_dots,
    spots_kwargs=dict(alpha=1.),
    ax=ax3
)
decoded_spots(
    dartfish_intensity_table,
    background_image=dartfish_nuclei_mp_numpy,
    spots_kwargs=dict(alpha=1.),
    ax=ax5
)
decoded_spots(
    decoded_image=dartfish_decoded_image,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax2
)
decoded_spots(
    decoded_image=dartfish_decoded_image,
    background_image=dartfish_dots,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax4
)
decoded_spots(
    decoded_image=dartfish_decoded_image,
    background_image=dartfish_nuclei_mp_numpy,
    decoded_image_kwargs=dict(alpha=1.),
    ax=ax6
);
# EPY: END code

# EPY: START markdown
### Show different background types for ISS
# EPY: END markdown

# EPY: START code
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, figsize=(30, 40))
decoded_spots(
    iss_intensity_table,
    background_image=np.zeros_like(iss_dots_mp_numpy),
    spots_kwargs=dict(alpha=1.),
    ax=ax1
)
decoded_spots(
    iss_intensity_table,
    background_image=iss_dots_mp_numpy,
    spots_kwargs=dict(alpha=1.),
    ax=ax3
)
decoded_spots(
    iss_intensity_table,
    background_image=iss_nuclei_mp_numpy,
    spots_kwargs=dict(alpha=1.),
    ax=ax5
)

# ISS doesn't have a decoded image right now, but we can make one! Leave the placeholders open.
for ax in (ax2, ax4, ax6):
    ax.set_axis_off()
f.tight_layout()
# EPY: END code

# EPY: START markdown
### Download available copy number information from assay authors
# EPY: END markdown

# EPY: START code
dartfish_copy_number = pd.read_csv(
    'https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH/fov_001/counts.csv',
    index_col=0,
    squeeze=True
)
merfish_copy_number = pd.read_csv(
    os.path.join(data_root, "merfish_copy_number_benchmark.csv"),
    index_col=0,
    squeeze=True
)
iss_copy_number = pd.read_csv(
    os.path.join(data_root, "iss_copy_number_benchmark.csv"),
    index_col=1
)['cnt']
# EPY: END code

# EPY: START code
f, axes = plt.subplots(ncols=3, nrows=4, figsize=(12, 14))
iterable = zip(
    [iss_intensity_table, merfish_intensity_table, dartfish_intensity_table],
    [iss_copy_number, merfish_copy_number, dartfish_copy_number],
    axes[0, :]
)
for dataset, benchmark, axis in iterable:
    compare_copy_number(dataset, benchmark, ax=axis, color='tab:blue')

for dataset, axis in zip(datasets, axes[1, :]):
    norms = dataset.feature_trace_magnitudes()
    histogram(norms, bins=20, log=True, ax=axis)

for dataset, axis in zip(datasets, axes[2, :]):
    area = (dataset.radius * np.pi) ** 2
    histogram(area, bins=20, ax=axis)

for dataset, axis in zip([merfish_intensity_table, dartfish_intensity_table], axes[3, 1:]):
    distances = dataset[Features.DISTANCE].values
    histogram(distances, bins=20, ax=axis)

# set the assay names as the titles of the top plots
axes[0, 0].set_title('In-Situ Sequencing', fontsize=20)
axes[0, 1].set_title('MERFISH', fontsize=20)
axes[0, 2].set_title('DARTFISH', fontsize=20);

# reset y-axis labels
for ax in np.ravel(axes):
    ax.set_ylabel('')

# reset titles
for ax in np.ravel(axes[1:, :]):
    ax.set_title('')

# set the y-axis labels
column_1_axes = (axes[:, 0])
plot_names = (
    'Copy Number Comparison\nwith Author Pipelines\n\nstarfish result\n\n',
    'Barcode Magnitude\nDistributions\n\nnumber of features',
    'Spot Area\nDistributions\n\nnumber of spots',
    'Feature Distances to\nNearest Code\n\nnumber of features\n\n'
)
for ax, name in zip(column_1_axes, plot_names):
    ax.set_ylabel(name, fontsize=16)


# fix up the figure
f.tight_layout()
for ax in np.ravel(axes):
    sns.despine(ax=ax)

# turn off the empty plot, matplotlib is super bad about this.
for ax in (axes[3, 0],):
    ax.xaxis.set_visible(False)
    # make spines (the box) invisible
    plt.setp(ax.spines.values(), visible=False)
    # remove ticks and labels for the left axis
    ax.tick_params(left=False, labelleft=False)
    # remove background patch (only needed for non-white background)
    ax.patch.set_visible(False)
# EPY: END code

# EPY: START markdown
#The histogram can be used to demonstrate parameter thresholding, as well.
# EPY: END markdown

# EPY: START code
f, ax = plt.subplots()
area = (iss_intensity_table.radius * np.pi) ** 2
histogram(
    area,
    bins=20,
    threshold=1000,
    title='fake threshold demonstration',
    ylabel='number of spots',
    xlabel='spot area'
);
# EPY: END code
