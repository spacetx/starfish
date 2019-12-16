.. _morphology:

Morphology Transformations
==========================

starfish provides a variety of methods to perform transformations on morphological data.  These include :py:class:`~starfish.morphology.Binarize`, which transform image data into morphological data and
:py:class:`~starfish.morphology.Filter`, which performs filtering operations on morphological data, and
:py:class:`~starfish.morphology.Merge`, which combines different sets of morphological data.

.. _binarize:

Binarize
--------

Binarizing operations can be imported using ``starfish.morphology.Binarize``, which registers all classes that subclass :py:class:`~starfish.morphology.Binarize.BinarizeAlgorithm`:

.. code-block:: python

    from starfish.morphology import Binarize

.. automodule:: starfish.morphology.Binarize
   :members:

.. _morphological_filter:

Filter
------

Filtering operations can be imported using ``starfish.morphology.Filter``, which registers all classes that subclass :py:class:`~starfish.morphology.Filter.FilterAlgorithm`:

.. code-block:: python

    from starfish.morphology import Filter

.. automodule:: starfish.morphology.Filter
   :members:

.. _merge:

Merge
-----

Filtering operations can be imported using ``starfish.morphology.Merge``, which registers all classes that subclass :py:class:`~starfish.morphology.Merge.MergeAlgorithm`:

.. code-block:: python

    from starfish.morphology import Merge

.. automodule:: starfish.morphology.Merge
   :members:
