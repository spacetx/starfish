.. _spots:

Spots
=====

Starfish provides a number of methods for which spots (or other regions of interest) are the main substrate.
These include :py:class:`starfish.spots.DetectPixels`, which exposes methods that identify which target code best corresponds to each pixel, and merges adjacent pixels into ROIs,
:py:class:`starfish.spots.FindSpots`, which exposes methods that find bright spots against dark backgrounds,
:py:class:`starfish.spots.DecodeSpots`, which exposes methods that match patterns of spots detected across rounds and channels in the same spatial positions with target codes, and
:py:class:`starfish.spots.AssignTargets`, which exposes methods to assign spots to cells.

.. _detect_pixels:

Detecting Pixels
----------------

Pixel Detectors can be imported using ``starfish.spots.DetectPixels``, which registers all classes that subclass ``DetectPixelsAlgorithm``:

.. code-block:: python

    from starfish.spots import DetectPixels

.. automodule:: starfish.spots.DetectPixels
    :members:

.. _spot_finding:

Finding Spots
---------------

Spot Finders can be imported using ``starfish.spots.FindSpots``, which registers all classes that subclass ``FindSpotsAlgorithm``:


.. _`Spot Finding Refactor Plan`: https://github.com/spacetx/starfish/issues/1514

.. code-block:: python

    from starfish.spots import FindSpots

.. automodule:: starfish.spots.FindSpots
    :members:

.. _decode_spots:

Decoding Spots
---------------

Spot Decoders can be imported using ``starfish.spots.DecodeSpots``, which registers all classes that subclass ``DecodeSpotsAlgorithm``:

.. code-block:: python

    from starfish.spots import DecodeSpots

.. automodule:: starfish.spots.DecodeSpots
   :members:


.. _target_assignment:

Target Assignment
-----------------

Target Assignment can be imported using ``starfish.spots.AssignTargets``, which registers all classes that subclass ``AssignTargetsAlgorithm``:

.. code-block:: python

    from starfish.spots import AssignTargets

.. automodule:: starfish.spots.AssignTargets
   :members:
