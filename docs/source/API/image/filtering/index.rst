.. _filtering:

Filtering
=========

Filters can be imported using ``starfish.image.Filter``, which registers all classes that subclass
``FilterAlgorithmBase``:

.. code-block:: python

    from starfish.image import Filter

.. contents::


Bandpass
--------

.. autoclass:: starfish.image._filter.bandpass.Bandpass
   :members:

Clip
----

.. autoclass:: starfish.image._filter.clip.Clip
    :members:

Gaussian High Pass
------------------

.. autoclass:: starfish.image._filter.gaussian_high_pass.GaussianHighPass
    :members:

Gaussian Low Pass
-----------------

.. autoclass:: starfish.image._filter.gaussian_low_pass.GaussianLowPass
    :members:

Mean High Pass
--------------

.. autoclass:: starfish.image._filter.mean_high_pass.MeanHighPass
    :members:

Deconvolve PSF
--------------

.. autoclass:: starfish.image._filter.richardson_lucy_deconvolution.DeconvolvePSF
    :members:

Scale By Percentile
-------------------

.. autoclass:: starfish.image._filter.scale_by_percentile.ScaleByPercentile
    :members:

White Top Hat
-------------

.. autoclass:: starfish.image._filter.white_tophat.WhiteTophat
    :members:

Zero By Channel Magnitude
-------------------------

.. autoclass:: starfish.image._filter.zero_by_channel_magnitude.ZeroByChannelMagnitude
    :members:
