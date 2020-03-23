"""
.. _howto_localmaxpeakfinder:

Finding Spots with :py:class:`.LocalMaxPeakFinder`
==================================================

RNA FISH spots are usually bright point spread functions in a greyscale image.
:term:`Rolonies<Rolony>`, which are rolling-circle amplicons produced in certain assays (e.g. in
situ sequencing), are approximately 1-um diameter Gaussian spots. Generally, the recommended
:py:class:`.FindSpotsAlgorithm` to use in a starfish pipeline is :py:class:`.BlobDetector`
because it accounts for the intensity profile of a spot rather than just thresholding pixel
values. But for some images :py:class:`.BlobDetector` may not be satisfactory so starfish also
provides alternatives.

:py:class:`.LocalMaxPeakFinder` finds spots by finding the optimal threshold to binarize image
into foreground (spots) and background and then using :py:func:`skimage.feature.peak_local_max` to
find local maxima or peaks in the foreground. Each peak is counted as a spot with radius equal to
one pixel. Using :py:class:`.LocalMaxPeakFinder` requires knowledge of expected spot sizes since it
uses ``min_obj_area`` and ``max_obj_area`` to filter foreground connected components before finding
peaks. ``min_distance`` is also used to to filter out noise by limiting the distance between peaks
that can be detected. The recommended way to set parameters is to take a representative image and
:ref:`visually assess <howto_spotfindingresults>` results.

.. warning::
    :py:class:`.LocalMaxPeakFinder` is not compatible with cropped data sets.

"""

# Load osmFISH experiment
from starfish import FieldOfView, data
from starfish.image import Filter
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, TraceBuildingStrategies
experiment = data.osmFISH(use_test_data=True)
imgs = experiment["fov_000"].get_image(FieldOfView.PRIMARY_IMAGES)

# filter raw data
filter_ghp = Filter.GaussianHighPass(sigma=(1,8,8), is_volume=True)
filter_laplace = Filter.Laplace(sigma=(0.2, 0.5, 0.5), is_volume=True)
filter_ghp.run(imgs, in_place=True)
filter_laplace.run(imgs, in_place=True)

# z project
max_imgs = imgs.reduce({Axes.ZPLANE}, func="max")

# run LocalMaxPeakFinder with dots as reference image
lmp = FindSpots.LocalMaxPeakFinder(
    min_distance=6,
    stringency=0,
    min_obj_area=6,
    max_obj_area=600,
    is_volume=True
)
spots = lmp.run(max_imgs)
