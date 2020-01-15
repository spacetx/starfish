.. _types:

Types
=====

.. contents::
    :local:

Starfish uses a series of constants and container types to enable type checking, and provides a set
of useful hints about how to interact with starfish objects.

Coordinates
-----------

Coordinates holds constants that store with the physical coordinates of a field of view. They define
a field of view's relative location to some global scale parameter, and identify how to stitch or
combine multiple fields of view.

.. autoclass:: starfish.core.types.Coordinates
    :members:
    :undoc-members:

Physical Coordinates
---------------------
.. autoclass:: starfish.core.types.PhysicalCoordinateTypes
    :members:
    :undoc-members:

Axes
----
Axes holds constants that represent indexers into the dimensions of the :py:class:`ImageStack`
5-d image tensor. They are re-used by objects that inherit subsets of these Axes, such as:

1. :py:class:`IntensityTable`, which stores spot coordinates and pixel traces across
   rounds and channels
2. :py:class:`Codebook`, which stores expected image intensities across imaging rounds and
   channels

.. autoclass:: starfish.core.types.Axes
    :members:
    :undoc-members:

Features
--------

Features holds constants that represent characteristics of detected image features (most often
spots, but sometimes also individual pixels).

.. autoclass:: starfish.core.types.Features
    :members:
    :undoc-members:


SpotAttributes
--------------

SpotAttributes defines the minimum amount of information required by starfish to describe
a spot. It also contains methods to save these attributes to files that can be used to visualize
detected spots.

.. autoclass:: starfish.core.types.SpotAttributes
    :members:

Levels
------

.. autoclass:: starfish.types.Levels
    :members:
    :undoc-members:
