.. _image:

Image Manipulation
==================


starfish provides a variety of image manipulation methods that aid in the quantification of image-based transcriptomics
experiments. These include :py:class:`~starfish.image.Filter`, which remove background fluorescence and enhance spots,
:py:class:`~starfish.image.LearnTransform`, which learn transforms to align images across rounds and channels,
:py:class:`~starfish.image.ApplyTransform`, which apply learned transforms to images, and finally,
:py:class:`~starfish.image.Segmentation`, to identify the locations of cells.


.. _filtering:

Filtering
---------

Filters can be imported using ``starfish.image.Filter``, which registers all classes that subclass
``FilterAlgorithm``:

.. code-block:: python

    from starfish.image import Filter

.. automodule:: starfish.image.Filter
   :members:


.. _learn_transform:

Learn Transform
---------------

LearnTransform can be imported using ``starfish.image.LearnTransform``, the subclasses of
``LearnTransformAlgorithm`` are available for transform learning.

.. code-block:: python

    from starfish.image import LearnTransform

.. automodule:: starfish.image.LearnTransform
   :members:


.. _apply_transform:

Apply Transform
---------------

ApplyTransform can be imported using ``starfish.image.ApplyTransform``, the subclasses of
``ApplyTransformAlgorithm`` are available for transform learning.

.. code-block:: python

    from starfish.image import ApplyTransform

.. automodule:: starfish.image.ApplyTransform
   :members:


.. _segmentation:

Segmentation
------------

Segmentation can be imported using ``starfish.image.Segment``, which registers all classes that subclass
``SegmentAlgorithm``:

.. code-block:: python

    from starfish.image import Segment

.. automodule:: starfish.image.Segment
   :members:
