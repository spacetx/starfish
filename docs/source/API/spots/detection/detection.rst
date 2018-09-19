.. _detection

Detecting
=========

Spot Detectors can be imported using ``starfish.spots.SpotFinder``, which registers all classes that
subclass ``SpotFinderAlgorithmBase``:

.. code-block:: python

    from starfish.spots import SpotFinder

.. contents::

Gaussian Spot Detector
----------------------

.. autoclass:: starfish.spots._detector.gaussian.GaussianSpotDetector
    :members:

Local Max Peak Finder
---------------------

.. autoclass:: starfish.spots._detector.local_max_peak_finder.LocalMaxPeakFinder
    :members:

Pixel Spot Detector
-------------------

.. autoclass:: starfish.spots._detector.pixel_spot_detector.PixelSpotDetector
    :members:
