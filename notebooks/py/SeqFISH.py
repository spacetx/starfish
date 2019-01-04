#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
import starfish.data
# EPY: END code

# EPY: START markdown
#This notebook demonstrates how to load SeqFISH data into starfish. Below loads a fraction of fov_000, a single field of view from the Intron-SeqFISH paper. We load the test data as the full dataset is ~ 5GB and may be difficult for some laptops to load. 
# EPY: END markdown

# EPY: START code
experiment = starfish.data.SeqFISH(use_test_data=True)
fov = experiment['fov_000']
stack = fov['primary']
# EPY: END code

# EPY: START code
stack.shape
# EPY: END code

# EPY: START markdown
#A processing pipeline is in the works. For now, you can explore the data in starfish. 
# EPY: END markdown
