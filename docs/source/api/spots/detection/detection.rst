.. _detection:

Detecting
=========

Spot Detectors can be imported using ``starfish.spots.DetectSpots``, which registers all classes that
subclass ``DetectSpotsAlgorithmBase``:

.. code-block:: python

    from starfish.spots import DetectSpots

.. contents::

Blob Detector
-------------

.. autoclass:: starfish.spots._detect_spots.blob.BlobDetector
    :members:

Local Max Peak Finder
---------------------

.. autoclass:: starfish.spots._detect_spots.local_max_peak_finder.LocalMaxPeakFinder
    :members:

Pixel Spot Decoder
-------------------

.. autoclass:: starfish.spots._detect_pixels.pixel_spot_decoder.PixelSpotDecoder
    :members:
