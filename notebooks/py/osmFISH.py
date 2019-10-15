#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Reproduce Published osmFISH results with Starfish
#
#osmFISH is an image based transcriptomics technique that can spatially resolve tens of RNA transcripts and their expression levels in-situ. The protocol anddata analysis are described in this [publication](https://www.nature.com/articles/s41592-018-0175-z). This notebook walks through how to use Starfish to process the raw images from an osmFISH experiment into a spatially resolved gene expression image. We verify taht Starfish can accurately reproduce the results from the authors' original Python [pipeline](http://linnarssonlab.org/osmFISH/image_analysis/)
#
#Please see [documentation](https://spacetx-starfish.readthedocs.io/en/stable/index.html) for detailed descriptions of all the data structures and methods used here.
# EPY: END markdown

# EPY: START code
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starfish import FieldOfView, data
from starfish.types import Axes

# EPY: ESCAPE %matplotlib inline
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: END code

# EPY: START markdown
### Load Data into Starfish from the Cloud
#
#The data from an osmFISH experiment are similar in form to a standard smFISH experiment. For each round, each color channel corresponds to presence of a particular gene. Across rounds, the color channels index different genes. Here, we analyze one FOV from the first round (r) channel (c) which consists of 45 z-planes (z). Each image in this image stack is of dimensions 2048x2048 (y X x). The data are taken from mouse somatosensory cortex, and the gene in this channel is Adloc
# EPY: END markdown

# EPY: START code
experiment = data.osmFISH(use_test_data=True)
imgs = experiment["fov_000"].get_image(FieldOfView.PRIMARY_IMAGES)
print(imgs)
# EPY: END code

# EPY: START markdown
### Filter and Visualize Data
# EPY: END markdown

# EPY: START markdown
#First, we remove background signal using a gaussian high-pass filter
# EPY: END markdown

# EPY: START code
from starfish.image import Filter

filter_ghp = Filter.GaussianHighPass(sigma=(1,8,8), is_volume=True)
imgs_ghp = filter_ghp.run(imgs, in_place=False)
# EPY: END code

# EPY: START markdown
#Next, we enhance the spots by filtering with a Laplace filter
# EPY: END markdown

# EPY: START code
filter_laplace = Filter.Laplace(sigma=(0.2, 0.5, 0.5), is_volume=True)
imgs_ghp_laplace = filter_laplace.run(imgs_ghp, in_place=False)
# EPY: END code

# EPY: START markdown
#Finally, we take a maximum projection over z, which effectively mitigates effects of out of focus z-planes
# EPY: END markdown

# EPY: START code
mp = imgs_ghp_laplace.reduce({Axes.ZPLANE}, func="max")
# EPY: END code

# EPY: START markdown
#We can now visualize our data before and after filtering
# EPY: END markdown

# EPY: START code
single_plane = imgs.reduce({Axes.ZPLANE}, func="max").xarray.sel({Axes.CH:0}).squeeze()
single_plane_filtered = mp.xarray.sel({Axes.CH: 0}).squeeze()

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(single_plane, cmap='gray', clim = list(np.percentile(single_plane.data, [1, 99.9])))
plt.axis('off')
plt.title('Original data, Round:0, Channel: 0')
plt.subplot(122)
plt.imshow(single_plane_filtered, cmap='gray', clim = list(np.percentile(single_plane_filtered.data, [1, 99.9])))
plt.title('Filtered data, Round:0, Channel: 0')
plt.axis('off');
# EPY: END code

# EPY: START markdown
### Decode the processed data into spatially resolved gene expression
# EPY: END markdown

# EPY: START markdown
#Decoding in a non-multiplexed image based transcriptomics method is equivalent to simple spot finding, since each spot in each color channel and round corresponds to a different gene. To find spots in osmFISH data, the authors employ a peak finder that distinguishes local maxima from their surroundings whose absolute intensities exceed a threshold value. It tests a number of different thresholds, building a curve from the number of peaks detected at each threshold. A threshold in the _stable region_ or _knee_ of the curve is selected, and final peaks are called with that threshold.
#
#This process is repeated independently for each round and channel. Here we show this process on a single round and channel to demonstrate the procedure. See the documentation for a precise description of the parameters.
# EPY: END markdown

# EPY: START code
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import TraceBuildingStrategies


lmp = FindSpots.LocalMaxPeakFinder(
    min_distance=6,
    stringency=0,
    min_obj_area=6,
    max_obj_area=600,
    is_volume=True
)
spots = lmp.run(mp)

decoder = DecodeSpots.PerRoundMaxChannel(codebook=experiment.codebook,
                                         trace_building_strategy=TraceBuildingStrategies.SEQUENTIAL)
decoded_intensities = decoder.run(spots=spots)
# EPY: END code

# EPY: START markdown
### Compare to pySMFISH peak calls
# EPY: END markdown

# EPY: START markdown
#The Field of view that we've used for the test data corresponds to Aldoc, imaged in round one, in position 33. We've also packaged the results from the osmFISH publication for this target to demonstrate that starfish is capable of recovering the same results.
# EPY: END markdown

# EPY: START code
def load_results(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def get_benchmark_peaks(loaded_results, redo_flag=False):

    if not redo_flag:
        sp = pd.DataFrame(
            {
                "y":loaded_results["selected_peaks"][:,0],
                "x":loaded_results["selected_peaks"][:,1],
                "selected_peaks_int": loaded_results["selected_peaks_int"],
            }
        )
    else:
        p = peaks(loaded_results)
        coords = p[p.thr_array==loaded_results["selected_thr"]].peaks_coords
        coords = coords.values[0]
        sp = pd.DataFrame({"x":coords[:,0], "y":coords[:,1]})

    return sp

try:
    module_path = __file__
except NameError:
    # this is probably being run from jupyter
    cwd = "."
else:
    cwd = os.path.dirname(module_path)
benchmark_results = load_results(os.path.join(
    cwd, "data", "EXP-17-BP3597_hyb1_Aldoc_pos_33.pkl"))
benchmark_peaks = get_benchmark_peaks(benchmark_results, redo_flag=False)
# EPY: END code

# EPY: START markdown
#Plot spots detected in the benchmark as blue spots, and overlay spots from starfish as orange x's. Starfish detects the same spot positions, but 41 fewer spots in total.
# EPY: END markdown

# EPY: START code
benchmark_spot_count = len(benchmark_peaks)
starfish_spot_count = len(decoded_intensities)

plt.figure(figsize=(10,10))
plt.plot(benchmark_peaks.x, -benchmark_peaks.y, "o")
plt.plot(decoded_intensities[Axes.X.value], -decoded_intensities[Axes.Y.value], "x")

plt.legend(["Benchmark: {} spots".format(benchmark_spot_count),
            "Starfish: {} spots".format(starfish_spot_count)])
plt.title("Starfish x osmFISH Benchmark Comparison");
# EPY: END code

# EPY: START code
spot_difference = benchmark_spot_count - starfish_spot_count
print(f"Starfish finds {spot_difference} fewer spots")
assert spot_difference == 41  # for starfish testing purposes
# EPY: END code
