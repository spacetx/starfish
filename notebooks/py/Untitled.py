#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
# EPY: ESCAPE %load_ext autoreload
# EPY: ESCAPE %autoreload 2
# EPY: END code

# EPY: START code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# EPY: ESCAPE %matplotlib notebook

import starfish
from starfish.types import Indices
from showit import image
# EPY: END code

# EPY: START code
experiment = starfish.Experiment.from_json('/Users/ambrosecarr/Desktop/MERFISH/experiment.json')
# EPY: END code

# EPY: START code
projection = experiment.image.max_proj(Indices.CH, Indices.ROUND, Indices.Z)
# EPY: END code

# EPY: START code
image(projection)
# EPY: END code

# EPY: START code
image(projection[950:1155, 545:950])
# EPY: END code

# EPY: START code
experiment.image._data = experiment.image._data[:, :, :, 950:1155, 545:950]

print('subset shape: ', experiment.image.shape)

experiment.image.write('/Users/ambrosecarr/Desktop/MERFISH_TEST/merfish')

# EPY: ESCAPE !ls /Users/ambrosecarr/Desktop/MERFISH_TEST/

assert np.array_equal(np.load('/Users/ambrosecarr/Desktop/MERFISH_TEST/merfish-X0-Y0-Z0-H0-C0.npy'), 
                      experiment.image.numpy_array[0, 0, 0, :, :])
# EPY: END code

# EPY: START code
print(np.load('/Users/ambrosecarr/Desktop/MERFISH_TEST/merfish-X0-Y0-Z0-H0-C0.npy').shape)
# EPY: END code
