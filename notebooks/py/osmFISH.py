#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
### Loading the data into Starfish
# EPY: END markdown

# EPY: START code
from starfish import data
from starfish.types import Indices
import os
# EPY: END code

# EPY: START markdown
#This notebook demonstrates how to load osmFISH data into starfish. Below loads fov_001, however fovs 002 and 003 are also converted and can be loaded by exchanging the number in the cloudflare link. The data can be dumped for local loading with `s.image.write`
# EPY: END markdown

# EPY: START code
use_test_data = os.getenv("USE_TEST_DATA") is not None
experiment = data.osmFISH(use_test_data=use_test_data)
# EPY: END code

# EPY: START markdown
#The below plot displays the z-volume for channel 0 of fov_001
# EPY: END markdown

# EPY: START code
image = experiment.fov().primary_image
image.show_stack({Indices.CH: 0})
# EPY: END code
