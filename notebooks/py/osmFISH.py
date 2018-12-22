#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float32
import pandas as pd

from starfish import ImageStack
from starfish.types import Indices
from starfish.image import Filter
from starfish.spots import SpotFinder
import starfish.data

# EPY: ESCAPE %matplotlib inline
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: END code

# EPY: START code
import starfish.data
# EPY: END code

# EPY: START code
experiment = starfish.data.osmFISH(use_test_data=True)
stack = experiment['fov_000']['primary']
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
def load_results(fov_num):
    pkls = glob(os.path.join(res_path, '*.pkl'))
    pkl = [p for p in pkls if str(fov_num) in p][0]
    with open(pkl, 'rb') as f:
        res = pickle.load(f)

    for k, v in res.items():
        if type(v) is np.float64 or type(v) is np.int64 or type(v) is np.int:
            print(k, v)

    return res

def selected_peaks(res, redo_flag = False):

    if not redo_flag:
        sp = pd.DataFrame({'y':res['selected_peaks'][:,0],
                           'x':res['selected_peaks'][:,1],
                           'selected_peaks_int': res['selected_peaks_int']
                          })
    else:
        p = peaks(res)
        coords = p[p.thr_array==res['selected_thr']].peaks_coords
        coords = coords.values[0]
        sp = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1]})

    return sp

def peaks(res):
    p = pd.DataFrame({'thr_array':res['thr_array'],
              'peaks_coords':res['peaks_coords'],
              'total_peaks':res['total_peaks']
             })
    return p

fov_num = 33
res_path = 'data'
res = load_results(fov_num)
sp = selected_peaks(res, redo_flag=False)
p = peaks(res)
psymFISH_thresh = res['selected_thr']
# EPY: END code

# EPY: START markdown
## Re-produce pysmFISH Results
# EPY: END markdown

# EPY: START markdown
### Filtering code
# EPY: END markdown

# EPY: START code
ghp = Filter.GaussianHighPass(sigma=(1,8,8), is_volume=True)
lp = Filter.Laplace(sigma=(0.2, 0.5, 0.5), is_volume=True)

stack_hp = ghp.run(stack, in_place=False)
stack_hp_lap = lp.run(stack_hp, in_place=False)
# EPY: END code

# EPY: START code
mp = stack_hp_lap.max_proj(Indices.Z)
for_vis = mp.xarray.sel({Indices.CH: 0}).squeeze()
# EPY: END code

# EPY: START code
plt.figure(figsize=(10,10))
plt.imshow(for_vis, cmap = 'gray', vmin=np.percentile(for_vis, 98), vmax=np.percentile(for_vis, 99.9))
plt.title('Filtered max projection')
plt.axis('off');
# EPY: END code

# EPY: START markdown
#### Spot Finding
# EPY: END markdown

# EPY: START code
min_distance = 6
stringency = 0
min_obj_area = 6
max_obj_area = 600

# TODO this will go away once ImageStack.max_proj returns an ImageStack
# stack = ImageStack.from_numpy_array(np.expand_dims(np.expand_dims(np.expand_dims(mp, 0), 0), 0))

lmp = SpotFinder.LocalMaxPeakFinder(
    min_distance=min_distance,
    stringency=stringency,
    min_obj_area=min_obj_area,
    max_obj_area=max_obj_area
)
lmp_res = lmp.run(mp)
# EPY: END code

# EPY: START markdown
#### Spot finding QA
# EPY: END markdown

# EPY: START code
plt.hist(lmp_res.data[:,0,0], bins=20)
plt.yscale('log')
plt.xlabel('Intensity')
plt.ylabel('Number of spots');
# EPY: END code

# EPY: START code
mp = stack_hp_lap.max_proj(Indices.Z)
mp = mp.sel({Indices.CH: 0, Indices.ROUND: 0}).xarray.squeeze()

plt.figure(figsize=(10,10))
plt.imshow(mp, cmap = 'gray', vmin=np.percentile(mp, 98), vmax=np.percentile(mp, 99.9))
plt.plot(lmp_res.x, lmp_res.y, 'or')
plt.axis('off');
# EPY: END code

# EPY: START markdown
### Compare to pySMFISH peak calls
# EPY: END markdown

# EPY: START code
num_spots_simone = len(sp)
num_spots_starfish = len(lmp_res)

plt.figure(figsize=(10,10))
plt.plot(sp.x, -sp.y, 'o')
plt.plot(lmp_res.x, -lmp_res.y, 'x')

plt.legend(['Benchmark: {} spots'.format(num_spots_simone),
            'Starfish: {} spots'.format(num_spots_starfish)])
plt.title('osmFISH spot calls')

print("Starfish finds {} fewer spots".format(num_spots_simone-num_spots_starfish))
# EPY: END code
