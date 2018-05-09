#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# # Reproduce Allen smFISH results with Starfish
# 
# The `allen_smFISH.zip` file needed to follow along with this notebook can be downloaded [here]()
# 
# This notebook walks through a work flow that reproduces the smFISH result for one field of view using the starfish package. 
# It assumes that you have unzipped `allen_smFISH.zip` in the same directory as this notebook. Thus, you should see:
# 
# raw/
# allen_smFISH.ipynb
# EPY: END markdown

# EPY: START code
import os
from starfish.io import Stack
import numpy as np
# EPY: END code

# EPY: START code
# package this up same as before
experiment_json = os.path.expanduser('~/google_drive/starfish/data/allen_smFISH/experiment.json')
# EPY: END code

# EPY: START code
s = Stack()
s.read(experiment_json)
# EPY: END code

# EPY: START code
# The allen's data is uint, but starfish loads it as floats. Cast it to int here, which does not cause any loss of precision. 
s.image._data = s.image._data.astype(np.uint16)
istack = s.image
# EPY: END code

# EPY: START code
# this doesn't work, do we want this abstraction? 
# s.image.dtype
# EPY: END code

# EPY: START markdown
# Image processing function list:
# 
# 1. clip & floor (at 10th percentile)
#   1. `starfish.transform.threshold` or `.clip`
#   2. Also called _after_ bandpass. 
# 2. trackpy bandpass
#   1. `filters.bandpass`
# 4. gaussian filter (over z) 
#   1. `starfish.filters.gaussian`. The skimage function supports nd images, so we just need to adjust the object in starfish. 
# 5. call peak locations (trackpy locate)
#   1. `starfish.starfish.spots.crocker-grier` OR `starfish.starfish.spots.gaussian` (second class in this file)
# EPY: END markdown
