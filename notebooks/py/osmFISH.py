#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import starfish
import starfish.data
from starfish import FieldOfView
from starfish.types import Axes

# EPY: ESCAPE %matplotlib inline
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: END code

# EPY: START code
experiment = starfish.data.osmFISH(use_test_data=True)
stack = experiment["fov_000"][FieldOfView.PRIMARY_IMAGES]
# EPY: END code

# EPY: START markdown
### Load pysmFISH results
# EPY: END markdown

# EPY: START markdown
#The Field of view that we've used for the test data corresponds to Aldoc, imaged in round one, in position 33. We've also packaged the results from the osmFISH publication for this target to demonstrate that starfish is capable of recovering the same results.
# EPY: END markdown

# EPY: START markdown
#The below commands parse and load the results from this file.
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
## Re-produce pysmFISH Results
# EPY: END markdown

# EPY: START markdown
### Filtering code
# EPY: END markdown

# EPY: START markdown
#Remove background using a gaussian high-pass filter, then enhance spots with a Laplacian filter.
# EPY: END markdown

# EPY: START code
filter_ghp = starfish.image.Filter.GaussianHighPass(sigma=(1,8,8), is_volume=True)
filter_laplace = starfish.image.Filter.Laplace(sigma=(0.2, 0.5, 0.5), is_volume=True)

stack_ghp = filter_ghp.run(stack, in_place=False)
stack_ghp_laplace = filter_laplace.run(stack_ghp, in_place=False)
# EPY: END code

# EPY: START markdown
#Max project over Z, then select the 1st `(0)` channel for visualization in the notebook to demonstrate the effect of background removal using these filters.
# EPY: END markdown

# EPY: START code
mp = stack_ghp_laplace.max_proj(Axes.ZPLANE)
array_for_visualization = mp.xarray.sel({Axes.CH: 0}).squeeze()
# EPY: END code

# EPY: START code
plt.figure(figsize=(10, 10))
plt.imshow(
    array_for_visualization,
    cmap="gray",
    vmin=np.percentile(array_for_visualization, 98),
    vmax=np.percentile(array_for_visualization, 99.9),
)
plt.title("Filtered max projection")
plt.axis("off");
# EPY: END code

# EPY: START markdown
#### Spot Finding
# EPY: END markdown

# EPY: START markdown
#osmFISH uses a peak finder that distinguishes local maxima from their surroundings whose absolute intensities exceed a threshold value. It tests a number of different thresholds, building a curve from the number of peaks detected at each threshold. A threshold in the _stable region_ of the curve is selected, and final peaks are called with that threshold.
#
#This process is repeated independently for each round and channel. Here we show this process on a single round and channel to demonstrate the procedure.
# EPY: END markdown

# EPY: START code
lmp = starfish.spots.SpotFinder.LocalMaxPeakFinder(
    min_distance=6,
    stringency=0,
    min_obj_area=6,
    max_obj_area=600,
)
spot_intensities = lmp.run(mp)
# EPY: END code

# EPY: START markdown
#### Spot finding QA
# EPY: END markdown

# EPY: START markdown
#Select spots in the first round and channel and plot their intensities
# EPY: END markdown

# EPY: START code
aldoc_spot_intensities = spot_intensities.sel({Axes.ROUND.value: 0, Axes.CH.value: 0})

plt.hist(aldoc_spot_intensities, bins=20)
plt.yscale("log")
plt.xlabel("Intensity")
plt.ylabel("Number of spots");
# EPY: END code

# EPY: START markdown
#Starfish enables maximum projection and slicing of the ImageStack object. However, these projections will maintain the 5d shape, leaving one-length dimensions for any array that has been projected over. Here the maximum projection of the z-plane of the ImageStack is calculated. From it, the first channel and round are selected, and `squeeze` is used to eliminate any dimensions with only one value, yielding a two-dimension `(x, y)` tile that can be plotted.
# EPY: END markdown

# EPY: START code
maximum_projection_5d = stack_ghp_laplace.max_proj(Axes.ZPLANE)
maximum_projection_2d = mp.sel({Axes.CH: 0, Axes.ROUND: 0}).xarray.squeeze()
# EPY: END code

# EPY: START markdown
#Use the maximum projection to plot all spots detected by starfish:
# EPY: END markdown

# EPY: START code
plt.figure(figsize=(10,10))
plt.imshow(
    maximum_projection_2d,
    cmap = "gray",
    vmin=np.percentile(maximum_projection_2d, 98),
    vmax=np.percentile(maximum_projection_2d, 99.9),
)
plt.plot(spot_intensities[Axes.X.value], spot_intensities[Axes.Y.value], "or")
plt.axis("off");
# EPY: END code

# EPY: START markdown
### Compare to pySMFISH peak calls
# EPY: END markdown

# EPY: START markdown
#Plot spots detected in the benchmark as blue spots, and overlay spots from starfish as orange x's. Starfish detects the same spot positions, but 41 fewer spots in total.
# EPY: END markdown

# EPY: START code
benchmark_spot_count = len(benchmark_peaks)
starfish_spot_count = len(spot_intensities)

plt.figure(figsize=(10,10))
plt.plot(benchmark_peaks.x, -benchmark_peaks.y, "o")
plt.plot(spot_intensities[Axes.X.value], -spot_intensities[Axes.Y.value], "x")

plt.legend(["Benchmark: {} spots".format(benchmark_spot_count),
            "Starfish: {} spots".format(starfish_spot_count)])
plt.title("Starfish x osmFISH Benchmark Comparison");
# EPY: END code

# EPY: START code
spot_difference = benchmark_spot_count - starfish_spot_count
print(f"Starfish finds {spot_difference} fewer spots")
assert spot_difference == 41  # for starfish testing purposes
# EPY: END code
