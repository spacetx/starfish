"""
.. _tutorial_diagnose_registration:

Diagnosing Registration
=======================

How to use :py:func:`.diagnose_registration` to visualize x and y alignment of images from
different rounds and channels of an :py:class:`.ImageStack`.

This example will use
"""

# Load ImageStack from example STARmap data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import starfish
import starfish.data
from starfish.types import Axes
from starfish.util.plot import diagnose_registration

experiment = starfish.data.STARmap(use_test_data=True)
stack = experiment['fov_000'].get_image('primary')

####################################################################################################
# Project ImageStack to only rounds... Finish later
ch_max_projection = stack.reduce({Axes.CH, Axes.ZPLANE}, func="max")

####################################################################################################
# The first three rounds are shown in blue, red, and green, respectively.
f, ax1 = plt.figure(dpi=150)
sel_0 = {Axes.ROUND: 0, Axes.X: (500, 600), Axes.Y: (500, 600)}
sel_1 = {Axes.ROUND: 1, Axes.X: (500, 600), Axes.Y: (500, 600)}
sel_2 = {Axes.ROUND: 2, Axes.X: (500, 600), Axes.Y: (500, 600)}
diagnose_registration(
    projection, sel_0, sel_1, sel_2, ax=ax1, title='diagnose registration'
)
