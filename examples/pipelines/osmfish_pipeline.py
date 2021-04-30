"""
.. _osmfish_example:

Reproduce published osmFISH results with starfish
=================================================

osmFISH is an image based transcriptomics technique that can spatially resolve tens of RNA
transcripts and their expression levels **in situ**. The protocol and data analysis are described in
this `publication`_. This notebook walks through how to use starfish to process the raw images
from an osmFISH experiment into a spatially resolved gene expression image. We verify that
starfish can accurately reproduce the results from the authors' original Python `pipeline`_

.. _publication: https://www.nature.com/articles/s41592-018-0175-z
.. _pipeline: http://linnarssonlab.org/osmFISH/image_analysis/
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
# Load Data into starfish from the cloud
# --------------------------------------
# The data from an osmFISH experiment are similar in form to a standard smFISH experiment. For each
# round, each color channel corresponds to presence of a particular gene. Across rounds, the color
# channels index different genes. Here, we analyze one FOV from the first round (r) and first
# channel (c), which consists of 45 z-planes (z). Each image in this image stack is of dimensions
# 2048x2048. The data are taken from mouse somatosensory cortex, and the gene in this channel is
# Adloc.

from starfish import data
from starfish import FieldOfView

experiment = data.osmFISH(use_test_data=True)
imgs = experiment["fov_000"].get_image(FieldOfView.PRIMARY_IMAGES)
print(imgs)

###################################################################################################
# Filter and visualize data
# -------------------------
# First, we remove background signal using a gaussian high-pass filter.

from starfish.image import Filter

filter_ghp = Filter.GaussianHighPass(sigma=(1, 8, 8), is_volume=True)
imgs_ghp = filter_ghp.run(imgs, in_place=False)

###################################################################################################
# Next, we enhance the spots by filtering with a Laplace filter.

filter_laplace = Filter.Laplace(sigma=(0.2, 0.5, 0.5), is_volume=True)
imgs_ghp_laplace = filter_laplace.run(imgs_ghp, in_place=False)

###################################################################################################
# Finally, we take a maximum projection over z, which effectively mitigates effects of out of focus
# z-planes.

from starfish.types import Axes

mp = imgs_ghp_laplace.reduce({Axes.ZPLANE}, func="max")

###################################################################################################
# We can now visualize our data before and after filtering.

import numpy as np

single_plane = imgs.reduce({Axes.ZPLANE}, func="max").xarray.sel({Axes.CH:0}).squeeze()
single_plane_filtered = mp.xarray.sel({Axes.CH: 0}).squeeze()

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(single_plane, cmap='gray', clim=list(np.percentile(single_plane.data, [1, 99.9])))
plt.axis('off')
plt.title('Original data, Round:0, Channel: 0')
plt.subplot(122)
plt.imshow(single_plane_filtered, cmap='gray', clim=list(np.percentile(single_plane_filtered.data, [1, 99.9])))
plt.title('Filtered data, Round:0, Channel: 0')
plt.axis('off')

###################################################################################################
# Decode the processed data into spatially resolved gene expression
# -----------------------------------------------------------------
# Decoding in a non-multiplexed image based transcriptomics method is equivalent to simple spot
# finding, since each spot in each color channel and round corresponds to a different gene. To
# find spots in osmFISH data, the authors employ a peak finder that distinguishes local maxima
# from their surroundings whose absolute intensities exceed a threshold value. It tests a number
# of different thresholds, building a curve from the number of peaks detected at each threshold.
# A threshold in the stable region or knee of the curve is selected, and final peaks are called
# with that threshold.
#
# This process is repeated independently for each round and channel. Here we show this process on
# a single round and channel to demonstrate the procedure. See the documentation for a precise
# description of the parameters.

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

###################################################################################################
# Compare to pySMFISH peak calls
# ------------------------------
# The Field of view that we've used for the test data corresponds to Aldoc, imaged in round one, in
# position 33. We've also packaged the results from the osmFISH publication for this target to
# demonstrate that starfish is capable of recovering the same results.

import os
import pandas as pd
import pickle


def load_results(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def get_benchmark_peaks(loaded_results, redo_flag=False):

    if not redo_flag:
        sp = pd.DataFrame(
            {
                "y":loaded_results["selected_peaks"][:, 0],
                "x":loaded_results["selected_peaks"][:, 1],
                "selected_peaks_int": loaded_results["selected_peaks_int"],
            }
        )
    else:
        p = peaks(loaded_results)
        coords = p[p.thr_array==loaded_results["selected_thr"]].peaks_coords
        coords = coords.values[0]
        sp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

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

###################################################################################################
# Plot spots detected in the benchmark as blue spots, and overlay spots from starfish as orange x's.
# Starfish detects the same spot positions, but 41 fewer spots in total.

benchmark_spot_count = len(benchmark_peaks)
starfish_spot_count = len(decoded_intensities)

plt.figure(figsize=(10, 10))
plt.plot(benchmark_peaks.x, -benchmark_peaks.y, "o")
plt.plot(decoded_intensities[Axes.X.value], -decoded_intensities[Axes.Y.value], "x")

plt.legend(["Benchmark: {} spots".format(benchmark_spot_count),
            "Starfish: {} spots".format(starfish_spot_count)])
plt.title("Starfish x osmFISH Benchmark Comparison")

spot_difference = benchmark_spot_count - starfish_spot_count
print(f"Starfish finds {spot_difference} fewer spots")
