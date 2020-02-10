"""
.. _tutorial_imshow_plane:

Showing Image of ImageStack Plane
=================================

How to use :py:func:`.imshow_plane` to display a single z-plane of an :py:class:`.ImageStack`.

As shown here, the :term:`selector<Selectors (Tile)>` parameter can pass a dictionary of
dimensions and indices to select a subset of the :py:class:`.ImageStack` as long as the subset is an
(x,y) plane. The selector can be omitted if :py:class:`.ImageStack` is already a single (x,y) plane.
"""

# Load ImageStack from example STARmap data
import starfish.data
from starfish import FieldOfView
from starfish.types import Axes
experiment = starfish.data.STARmap(use_test_data=True)
stack = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print(stack)

# Maximum project ImageStack along z-axis
projection = stack.reduce({Axes.ZPLANE}, func="max")
print(projection)

# Plot
import matplotlib
import matplotlib.pyplot as plt
from starfish.util.plot import imshow_plane
matplotlib.rcParams["figure.dpi"] = 150
f, (ax1, ax2, ax3) = plt.subplots(ncols=3)

# Plot first round and channel of projected ImageStack
imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 1, Axes.ZPLANE: 10}, ax=ax1, title='Z: 10')
imshow_plane(projection, sel={Axes.ROUND: 0, Axes.CH: 1}, ax=ax2, title='Max Projection')
# Plot ROI of projected image
selector = {Axes.CH: 0, Axes.ROUND: 0, Axes.X: (400, 600), Axes.Y: (550, 750)}
imshow_plane(projection, sel=selector, ax=ax3, title='Max Projection\nROI')

