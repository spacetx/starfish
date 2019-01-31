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

.. literalinclude:: /../../starfish/types/_constants.py
    :pyobject: Coordinates

Axes
----
Axes holds constants that represent indexers into the dimensions of the :py:class:`ImageStack` 
5-d image tensor. They are re-used by objects that inherit subsets of these Axes, such as:

1. :py:class:`IntensityTable`, which stores spot coordinates and pixel traces across 
   rounds and channels
2. :py:class:`Codebook`, which stores expected image intensities across imaging rounds and 
   channels

.. literalinclude:: /../../starfish/types/_constants.py
    :pyobject: Coordinates

Features
--------

Features holds constants that represent characteristics of detected image features (most often 
spots, but sometimes also individual pixels). 

.. literalinclude:: /../../starfish/types/_constants.py
    :pyobject: Coordinates


SpotAttributes
--------------

SpotAttributes defines the minimum amount of information required by starfish to describe 
a spot. It also contains methods to save these attributes to files that can be used to visualize
detected spots. 

.. literalinclude:: /../../starfish/types/_spot_attributes.py
    :pyobject: SpotAttributes
