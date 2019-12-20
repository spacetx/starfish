"""
Feature Identification
======================

"""

####################################################################################################
# .. _tutorial_spot_finding:
#
# Spot Finding
# ============
#
# Starfish's spot finding approaches are in the process of being revised. At the moment, *starfish*
# defines *more* than the number of spot detectors are required, exposing a number of different
# iterations of local blob detectors, and a pixel-based spot detector. In the future, the pixel
# decoder will move to become a filter, and the spot detectors will be collapsed into a single method
# that integrates the various automation present in the disparate spot detectors.
#
# However, for the time being, Starfish exposes a number of spot detectors with the following
# characteristics:
#
# =========================  =============  ===============  ============  =================
# Method                     Works in 3d    finds threshold  finds sigma   anisotropic sigma
# -------------------------  -------------  ---------------  ------------  -----------------
# BlobDetector               |yes|          |no|             |yes|         |yes|
# LocalMaxPeakFinder         |yes|          |yes|            |no|          |no|
# TrackpyLocalMaxPeakFinder  |yes|          |no|             |no|          |yes|
# =========================  =============  ===============  ============  =================
#
# .. |yes| unicode:: U+2705 .. White Heavy Check Mark
# .. |no| unicode:: U+274C .. Cross Mark
#
# BlobDetector and LocalMaxPeakFinder should usually be chosen over TrackpyLocalMaxPeakFinder, and
# BlobDetector should be favored over LocalMaxPeakFinder if you are unsure of the size of the spot and
# the spots are uniformly gaussian in shape. LocalMaxPeakFinder, by contrast, can help find the
# correct threshold.
#

pass

####################################################################################################
# .. _tutorial_spot_decoding:
#
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
#

pass

####################################################################################################
# .. _tutorial_segmenting_cells:
#
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
#
# TODO cell segmentation demo
#
pass

####################################################################################################
# .. _tutorial_assigning_spots_to_cells:
#
# Assigning Spots to Cells
# ========================
#
# TODO cell assignment demo and creating Cell x Gene matrix

pass
