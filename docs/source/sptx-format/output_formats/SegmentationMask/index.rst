.. _SegmentationMask:

SegmentationMask
================
The :py:class:`SegmentationMask` is an integer array the same size as the :code:`x-y` plane of the
input :py:class:`ImageStack`. The values of the :py:class:`SegmentationMask` indicate which cell
each pixel corresponds to. At the moment, only hard assignment is supported.

Implementation
--------------
The :py:class:`SegmentationMask` is currently implemented as a :code:`numpy.ndarray` object. We are
revisiting the optimal object for this information, and are aware of the pending need to use
segmented instances of cells to label points in 3d, and also to support non-cell instances. The
current implementation is expected to change.

Serialization
-------------
numpy arrays can be saved in a variety of formats of which the most common is npy_

.. _npy: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.save.html