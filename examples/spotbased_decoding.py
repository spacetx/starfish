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
minimum peak intensity threshold.

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


