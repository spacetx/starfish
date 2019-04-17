.. _detection:

Detecting
=========

Spot Detectors can be imported using ``starfish.spots.SpotFinder``, which registers all classes that
subclass ``SpotFinderAlgorithmBase``:

.. code-block:: python

    from starfish.spots import SpotFinder

.. contents::

Blob Detector
-------------

.. autoclass:: starfish.spots._detector.blob.BlobDetector
    :members:

Local Max Peak Finder
---------------------

.. autoclass:: starfish.spots._detector.local_max_peak_finder.LocalMaxPeakFinder
    :members:

Pixel Spot Decoder
-------------------

.. autoclass:: starfish.spots._pixel_decoder.pixel_spot_decoder.PixelSpotDecoder
    :members:
