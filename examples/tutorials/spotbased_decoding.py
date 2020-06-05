"""
.. _tutorial_spot_based_decoding:

Spot-Based Decoding
===================

Spot-based decoding is the approach of finding spots in images from each round first and then
decoding them. The alternative, pixel-based decoding, decodes pixels first and then connects them
into spots after.

.. image:: /_static/design/decoding_flowchart.png
   :scale: 50 %
   :alt: Decoding Flowchart
   :align: center

Starfish provides multiple options for each component of spot-based decoding:

Spot Finding
------------

.. list-table:: :py:class:`.FindSpotsAlgorithm`
   :widths: auto
   :header-rows: 1

   * - Method
     - Description
     - Works in 3D
     - Finds Threshold
     - Finds Sigma
     - Anisotropic Sigma
   * - :py:class:`.BlobDetector`
     - Wrapper of classic kernel convolution blob detection algorithms in :py:mod:`skimage.feature`
       such as LoG, which uses the Laplacian of Gaussian filter
     - |yes|
     - |no|
     - |yes|
     - |yes|
   * - :py:class:`.LocalMaxPeakFinder`
     - Wrapper of :py:mod:`skimage.feature.peak_local_max`, which finds local maxima pixel
       intensities in an image
     - |yes|
     - |yes|
     - |no|
     - |no|
   * - :py:class:`.TrackpyLocalMaxPeakFinder`
     - Wrapper for :py:mod:`trackpy.locate`, which implements a version of the Crocker-Grier
       algorithm originally developed for particle tracking
     - |yes|
     - |no|
     - |no|
     - |yes|

:py:class:`.BlobDetector` and :py:class:`.LocalMaxPeakFinder` should usually be chosen over
:py:class:`.TrackpyLocalMaxPeakFinder`, and :py:class:`.BlobDetector` should be favored over
:py:class:`.LocalMaxPeakFinder` if you are unsure of the size of the spot and the spots are
uniformly gaussian in shape. :py:class:`.LocalMaxPeakFinder`, by contrast, can help find the correct
minimum peak intensity threshold. A demonstration of their differences can be found at the end of
this tutorial.

Detected spots are returned in :py:class:`.SpotFindingResults`, which can be
:ref:`visually assessed <howto_spotfindingresults>` before decoding.

* :ref:`howto_blobdetector`
* :ref:`howto_localmaxpeakfinder`
* :ref:`howto_trackpylocalmaxpeakfinder`

Trace Building
--------------

.. list-table:: ``TraceBuildingStrategy``
   :widths: auto
   :header-rows: 1

   * - Method
     - Description
     - Reference Image
   * - ``SEQUENTIAL``
     - Build traces for every detected spot by setting intensity values to zero for all rounds
       and channels the spot was not found in (i.e. every trace will have only 1 non-zero value)
     - Incompatible
   * - ``EXACT_MATCH``
     - Build traces by combining intensity values of spots from every rounds and channel in the
       exact same location as spots in ``reference_image``
     - Required
   * - ``NEAREST_NEIGHBOR``
     - Build traces by combining intensity values of spots from rounds and channels nearest to the
       spots in the ``anchor_round``
     - Not recommended; will have same result as EXACT_MATCH

The first step to decoding :py:class:`.SpotFindingResults` is identifying spots from
different imaging rounds as the same spot. In starfish this is referred to as building traces and it
transforms :py:class:`.SpotFindingResults` to an :py:class:`.IntensityTable`. Trace building is
hidden in the :py:class:`.DecodeSpotsAlgorithm` but it requires the user to select a
``TraceBuildingStrategy``. :ref:`howto_tracebuildingstrategies`. goes further in depth and shows
how to build traces independent of the decoding step.

.. _spot_decoding_table:

Spot Decoding
-------------

.. list-table:: :py:class:`.DecodeSpotsAlgorithm`
   :widths: auto
   :header-rows: 1

   * - Method
     - Description
     - Works in 3D
     - Codebook Design
     - TraceBuildingStrategy
     - Returns Quality Score
   * - :py:class:`.SimpleLookupDecoder`
     - Description
     - |yes|
     - Linearly multiplexed
     - Sequential
     - |no|
   * - :py:class:`.PerRoundMaxChannel`
     - Description
     - |yes|
     - One hot exponentially multiplexed
     - Sequential, Exact_Match or Nearest_Neighbor
     - |yes|
   * - :py:class:`.MetricDistance`
     - Description
     - |yes|
     - Exponentially multiplexed
     - Exact_Match or Nearest_Neighbor
     - |yes|

.. |yes| unicode:: U+2705 .. White Heavy Check Mark
.. |no| unicode:: U+274C .. Cross Mark

Starfish decoding is done by running a :py:class:`.DecodeSpotsAlgorithm` on
:py:class:`.SpotFindingResults` to return a :py:class:`.DecodedIntensityTable`.
:py:class:`.PerRoundMaxChannel` should generally be used rather than the
the other two decoding algorithms if possible. :py:class:`.MetricDistance` is necessary for
:term:`codebooks<Codebook>` that contain :term:`codewords<Codeword>` without exactly one hot
channel in every round. This is used for error-robustness (e.g. MERFISH) and/or reducing optical
crowding in each round.

* :ref:`howto_simplelookupdecoder`
* :ref:`howto_perroundmaxchannel`
* :ref:`howto_metricdistance`

"""

####################################################################################################
# Comparison of :py:class:`.FindSpotsAlgorithm`\s
# -----------------------------------------------
# This tutorial demonstrates usage and characteristics of the three available
# :py:class:`.FindSpotsAlgorithm`\s on 3-Dimensional STARmap images. Parameters were roughly
# tuned, but these results are not reflective of the best possible performance of each
# :py:class:`.FindSpotsAlgorithm`.

# load STARmap data
from starfish import data, display
from starfish.image import ApplyTransform, LearnTransform
from starfish.spots import FindSpots
from starfish.types import Axes
from starfish.util.plot import imshow_plane
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match

experiment = data.STARmap(use_test_data=True)
imgs = experiment['fov_000'].get_image('primary')

# register rounds
projection = imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max")
reference_image = projection.sel({Axes.ROUND: 0})
ltt = LearnTransform.Translation(reference_stack=reference_image, axes=Axes.ROUND, upsampling=1000)
transforms = ltt.run(projection)
warp = ApplyTransform.Warp()
imgs = warp.run(stack=imgs, transforms_list=transforms)

# make reference round for decoding
dots = imgs.reduce({Axes.CH, Axes.ROUND}, func="max")

# view a cropped region of image for spot finding
imshow_plane(dots, sel={Axes.ZPLANE: 15,  Axes.X: (400, 600),  Axes.Y: (400, 600)})

####################################################################################################
# The ``dots`` reference image shows spots are approximately 10 pixels in diameter and can be
# tightly packed together. This helped inform the parameter settings of the spot finders below.
# However, the accuracy can always be improved by further tuning parameters. For example,
# if low intensity background noise is being detected as spots, increasing values for
# ``threshold``, ``stringency``, ``min_mass``, and ``percentile`` can remove them. If large spots
# are not missed, increasing values such as ``max_sigma``, ``max_obj_area``, ``spot_diameter``,
# and ``max_size`` could include them. Moreover, signal enhancement and background reduction
# prior to this step can also improve accuracy of spot finding.

bd = FindSpots.BlobDetector(
    min_sigma=2,
    max_sigma=6,
    num_sigma=20,
    threshold=0.1,
    is_volume=True,
    measurement_type='mean',
)

lmp = FindSpots.LocalMaxPeakFinder(
    min_distance=2,
    stringency=8,
    min_obj_area=6,
    max_obj_area=600,
    is_volume=True
)

tlmpf = FindSpots.TrackpyLocalMaxPeakFinder(
    spot_diameter=11,
    min_mass=0.2,
    max_size=8,
    separation=3,
    preprocess=False,
    percentile=80,
    verbose=True,
)

# crop imagestacks
crop_selection = {Axes.X: (300, 700), Axes.Y: (300, 700)}
cropped_imgs= imgs.sel(crop_selection)
cropped_dots = dots.sel(crop_selection)

# find spots on cropped images
bd_spots = bd.run(image_stack=cropped_imgs, reference_image=cropped_dots)
lmp_spots = lmp.run(image_stack=cropped_imgs, reference_image=cropped_dots)
tlmpf_spots = tlmpf.run(image_stack=cropped_imgs, reference_image=cropped_dots)

# build spot traces into intensity table
bd_table = build_spot_traces_exact_match(bd_spots)
lmp_table = build_spot_traces_exact_match(lmp_spots)
tlmpf_table = build_spot_traces_exact_match(tlmpf_spots)

# plot spots found
import matplotlib
import matplotlib.pyplot as plt

# get x, y coords and spot size from intensity table
def get_cropped_coords(table, x_min, x_max, y_min, y_max):
    df = table.to_features_dataframe()
    df = df.loc[df['x'].between(x_min, x_max) & df['y'].between(y_min, y_max)]
    return df['x'].values-x_min, df['y'].values-y_min, df['radius'].values.astype(int)
bd_x, bd_y, bd_s = get_cropped_coords(bd_table, 200, 250, 200, 250)
lmp_x, lmp_y, lmp_s = get_cropped_coords(lmp_table, 200, 250, 200, 250)
tlmpf_x, tlmpf_y, tlmpf_s = get_cropped_coords(tlmpf_table, 200, 250, 200, 250)

matplotlib.rcParams["figure.dpi"] = 150
f, (ax1, ax2, ax3) = plt.subplots(ncols=3)

# Plot cropped region of max z-projected dots image
imshow_plane(cropped_dots.reduce({Axes.ZPLANE}, func="max"), sel={Axes.X: (200, 250), Axes.Y: (200, 250)}, ax=ax1, title='BlobDetector')
imshow_plane(cropped_dots.reduce({Axes.ZPLANE}, func="max"), sel={Axes.X: (200, 250), Axes.Y: (200, 250)}, ax=ax2, title='LocalMaxPeak')
imshow_plane(cropped_dots.reduce({Axes.ZPLANE}, func="max"), sel={Axes.X: (200, 250), Axes.Y: (200, 250)}, ax=ax3, title='Trackpy')
# Overlay spots found by each FindSpotsAlgorithm
ax1.scatter(bd_x, bd_y, marker='o', facecolors='none', edgecolors='r', s=bd_s*10)
ax2.scatter(lmp_x, lmp_y, marker='o', facecolors='none', edgecolors='r', s=lmp_s*10)
ax3.scatter(tlmpf_x, tlmpf_y, marker='o', facecolors='none', edgecolors='r', s=tlmpf_s*10)

####################################################################################################
# These images show :py:class:`.BlobDetector` is good at finding round Gaussian spots with
# separation from one another. However, multiple spots packed closely together may be detected as
# one large spot. :py:class:`.LocalMaxPeakFinder` and :py:class:`.TrackpyLocalMaxPeakFinder` are
# more sensitive to peaks of intensity, which allows them to detect multiple partially
# overlapping spots but also makes them more vulnerable to noise. Smoothing the image with a low
# band pass filter can mitigate that issue. The last thing to note is spots found by
# :py:class:`.LocalMaxPeakFinder` all have a radius of one, because unlike the other two
# algorithms it isn't technically finding spots but just peaks of intensity.
#
# The multi-dimensional spot finding results can also be viewed interactively with napari using
# the code below:

# uncomment to view with napari
# %gui qt
# BlobDetector
# viewer = display(stack=cropped_dots, spots=bd_table)

# LocalMaxPeakFinder
# display(stack=cropped_dots, spots=lmp_table)

# TrackpyLocalMaxPeakFinder
# display(stack=cropped_dots, spots=tlmpf_table)