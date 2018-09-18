.. _filtering

Filtering
=========

Filters can be imported using ``starfish.image.Filter``, which registers all classes that subclass
``FilterAlgorithmBase``:

.. code-block:: python

    from starfish.image import Filter

.. contents::


Bandpass
--------

.. autoclass:: starfish.image.Filter.Bandpass
   :members:

Clip
----

.. autoclass:: starfish.image.Filter.Clip
    :members:

Gaussian High Pass
------------------

.. autoclass:: starfish.image.Filter.GaussianHighPass
    :members:

Gaussian Low Pass
-----------------

.. autoclass:: starfish.image.Filter.GaussianLowPass
    :members:

Mean High Pass
--------------

.. autoclass:: starfish.image.Filter.MeanHighPass
    :members:

Deconvolve PSF
--------------

.. autoclass:: starfish.image.Filter.DeconvolvePSF
    :members:

Scale By Percentile
-------------------

.. autoclass:: starfish.image.Filter.ScaleByPercentile
    :members:

White Top Hat
-------------

.. autoclass:: starfish.image.Filter.WhiteTophat
    :members:

Zero By Channel Magnitude
-------------------------

.. autoclass:: starfish.image.Filter.ZeroByChannelMagnitude
    :members:
