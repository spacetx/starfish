.. _spots:

Spots
=====

Starfish provides a number of methods for which spots (or other regions of interest) are the main substrate.
These include :py:class:`starfish.spots.DetectPixels`, which exposes methods that identify which target code best corresponds to each pixel, and merges adjacent pixels into ROIs,
:py:class:`starfish.spots.DetectSpots`, which exposes methods that find bright spots against dark backgrounds,
:py:class:`starfish.spots.Decode`, which exposes methods that match patterns of spots detected across rounds and channels in the same spatial positions with target codes, and
:py:class:`starfish.spots.AssignTargets`, which exposes methods to assign spots to cells.

.. _detect_pixels:

Detecting Pixels
----------------

Pixel Detectors can be imported using ``starfish.spots.DetectPixels``, which registers all classes that subclass ``DetectPixelsAlgorithm``:

.. code-block:: python

    from starfish.spots import DetectPixels

.. autoclass:: starfish.spots.DetectPixels
    :members:


.. _detection:

Detecting Spots
---------------

Spot Detectors can be imported using ``starfish.spots.DetectSpots``, which registers all classes that subclass ``DetectSpotsAlgorithm``:

.. code-block:: python

    from starfish.spots import DetectSpots

.. autoclass:: starfish.spots.DetectSpots
    :members:


.. _decoding:

Decoding
--------

Decoders can be imported using ``starfish.spots.Decode``, which registers all classes that subclass ``DecodeAlgorithm``:

.. code-block:: python

    from starfish.spots import Decode

.. autoclass:: starfish.spots.Decode
   :members:


.. _target_assignment:

Target Assignment
-----------------

Target Assignment can be imported using ``starfish.spots.AssignTargets``, which registers all classes that subclass ``AssignTargetsAlgorithm``:

.. code-block:: python

    from starfish.spots import AssignTargets

.. autoclass:: starfish.spots.AssignTargets
   :members:
