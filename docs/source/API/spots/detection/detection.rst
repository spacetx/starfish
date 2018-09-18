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

.. autoclass:: starfish.spots.SpotFinder.GaussianSpotDetector
    :members:

Local Max Peak Finder
---------------------

.. autoclass:: starfish.spots.SpotFinder.LocalMaxPeakFinder
    :members:

Pixel Spot Detector
-------------------

.. autoclass:: starfish.spots.SpotFinder.PixelSpotDetector
    :members:
