"""
Spot Finding
============

Starfish's spot finding approaches are in the process of being revised. At the moment, *starfish*
defines *more* than the number of spot detectors are required, exposing a number of different
iterations of local blob detectors, and a pixel-based spot detector. In the future, the pixel
decoder will move to become a filter, and the spot detectors will be collapsed into a single method
that integrates the various automation present in the disparate spot detectors.

However, for the time being, Starfish exposes a number of spot detectors with the following
characteristics:

=========================  =============  ===============  ============  =================
Method                     Works in 3d    finds threshold  finds sigma   anisotropic sigma
-------------------------  -------------  ---------------  ------------  -----------------
BlobDetector               |yes|          |no|             |yes|         |yes|
LocalMaxPeakFinder         |yes|          |yes|            |no|          |no|
TrackpyLocalMaxPeakFinder  |yes|          |no|             |no|          |yes|
=========================  =============  ===============  ============  =================

.. |yes| unicode:: U+2705 .. White Heavy Check Mark
.. |no| unicode:: U+274C .. Cross Mark

BlobDetector and LocalMaxPeakFinder should usually be chosen over TrackpyLocalMaxPeakFinder, and
BlobDetector should be favored over LocalMaxPeakFinder if you are unsure of the size of the spot and
the spots are uniformly gaussian in shape. LocalMaxPeakFinder, by contrast, can help find the
correct threshold.

First, the background must be removed.
"""

import starfish
from starfish import FieldOfView, data
from starfish.image import Filter
from starfish.types import Axes

experiment = data.allen_smFISH(use_test_data=True)

fov = experiment.fov()

imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES)
nuclei = fov.get_image('nuclei')

# bandpass filter to remove cellular background and camera noise
bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)

# gaussian blur to smooth z-axis
glp = Filter.GaussianLowPass(
    sigma=(1, 0, 0),
    is_volume=True
)

# pre-filter clip to remove low-intensity background signal
clip1 = Filter.Clip(p_min=50, p_max=100)

# post-filter clip to eliminate all but the highest-intensity peaks
clip2 = Filter.Clip(p_min=99, p_max=100, is_volume=True)

filter_kwargs = dict(
    in_place=True,
    verbose=True,
)
clip1.run(imgs, **filter_kwargs)
bandpass.run(imgs, **filter_kwargs)
glp.run(imgs, **filter_kwargs)
clip2.run(imgs, **filter_kwargs)

###################################################################################################
# Then the spot-finding algorithm can be run.

from starfish.spots import SpotFinder

# peak caller
tlmpf = SpotFinder.TrackpyLocalMaxPeakFinder(
    spot_diameter=5,
    min_mass=0.02,
    max_size=2,
    separation=7,
    noise_size=0.65,
    preprocess=False,
    percentile=10,
    verbose=True,
    is_volume=True,
)

spot_attributes = tlmpf.run(imgs)

###################################################################################################
# Spot Decoding
# =============
#
# Starfish's decoding methods are based on its :ref:`Codebook` class. There are two primary decoding
# methods exposed by starfish, a metric based decoder, and a per-channel max decoder.
#
# Metric Decoder
# --------------
# The metric decoder is a method associated with a codebook object. It normalizes both the codes and
# the intensities across the codebook to unit length using any L_p norm. It then applies any metric
# available in `scipy.spatial.distance`_.
#
# .. _scipy.spatial.distance: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
#
# Per Channel Max Decoder
# -----------------------
#
# The :ref:`Codebook.per_channel_max_decode <Codebook>` is a specialized decoder that takes advantage of
# codebooks that are designed to fluoresce in exactly one channel per round. Many image-based
# transcriptomics approaches use four rounds to read out the four DNA nucleotides, and as such
# this decoding method is a common one.

codebook = experiment.codebook
decoded = codebook.decode_per_round_max(spot_attributes)
decoded = decoded[decoded['total_intensity'] > .025]

###################################################################################################
# Segmenting Cells
# ================
#
# Cell segmentation is a very challenging task for image-based transcriptomics experiments. There
# are many approaches that do a very good job of segmenting *nuclei*, but few if any automated
# approaches to segment *cells*. Starfish exposes the watershed segmentation method from classical
# image processing, which inverts the intensities of the nuclei and spots and treats them like a
# literal watershed basin. The method then sets a threshold (water line), and each basin is treated
# as its own separate segment (cell).
#
# This approach works fairly well for cultured cells and relatively sparse tissues, but often cannot
# segment denser epithelia. As such, starfish *also* defines a simple segmentation format to enable it
# to read and utilize segmentation results derived from hand-drawing or semi-supervised drawing
# applications.

from starfish.image import Segmentation

dapi_thresh = .3     # binary mask for cell (nuclear) locations
stain_thresh = .035  # binary mask for overall cells // binarization of stain
min_dist = 10

nuclei = nuclei.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)


seg = Segmentation.Watershed(
    nuclei_threshold=dapi_thresh,
    input_threshold=stain_thresh,
    min_distance=min_dist
)
masks = seg.run(imgs, nuclei)
seg.show()

###################################################################################################
# Assigning Spots to Cells
# ========================
#
# Starfish assigns spots to cells via a simple check if each spot's centroid
# falls within a segmented cell.

from starfish.spots import TargetAssignment

al = TargetAssignment.Label()
labeled = al.run(masks, decoded)
cg = labeled.to_expression_matrix()
print(cg)
