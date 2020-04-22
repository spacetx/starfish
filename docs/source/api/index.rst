.. _API:

API Reference
=============

.. _image:

Image Manipulation
------------------

starfish provides a variety of image manipulation methods that aid in the quantification of image-based transcriptomics
experiments. These include :py:class:`~starfish.image.Filter`, which remove background fluorescence and enhance spots,
:py:class:`~starfish.image.LearnTransform`, which learn transforms to align images across rounds and channels,
:py:class:`~starfish.image.ApplyTransform`, which apply learned transforms to images, and finally,
:py:class:`~starfish.image.Segmentation`, to identify the locations of cells.


.. autosummary::

   starfish.image.Filter
   starfish.image.LearnTransform
   starfish.image.ApplyTransform
   starfish.image.Segment

.. toctree::
   :hidden:

   image/index

.. toctree::
   data_structures/index

.. toctree::
   spots/index.rst

.. toctree::
   morphology/index.rst

.. toctree::
   types/index.rst

.. toctree::
   validation/index.rst

.. toctree::
   utils/index.rst

.. toctree::
   datasets/index.rst
