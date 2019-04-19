.. _segmentation:

Segmentation
============

Segmentation can be imported using ``starfish.image.Segment``, which registers all classes that subclass
``SegmentAlgorithmBase``:

.. code-block:: python

    from starfish.image import Segment

.. contents::


Watershed
---------

.. autoclass:: starfish.image._segment.watershed.Watershed
   :members:

