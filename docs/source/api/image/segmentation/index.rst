.. _segmentation:

Segmentation
============

Segmentation can be imported using ``starfish.image.Segmentation``, which registers all classes
that subclass ``SegmentationAlgorithmBase``:

.. code-block:: python

    from starfish.image import Segmentation

.. contents::


Watershed
---------

.. autoclass:: starfish.image._segmentation.watershed.Watershed
   :members:

