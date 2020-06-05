.. _experiment_builder:

Data Formatting
===============

Classes and functions for converting data to SpaceTx Format and building experiments.

Converting Structured Data
--------------------------

:ref:`format_structured_data` tutorial.

.. code-block:: python

    from starfish.experiment.builder import format_structured_dataset

.. automodule:: starfish.core.experiment.builder.structured_formatter
   :members: format_structured_dataset

Tile Fetcher Interface
----------------------

:ref:`format_tilefetcher` tutorial and :ref:`tilefetcher_loader` tutorial.

.. code-block:: python

    from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json

.. automodule:: starfish.core.experiment.builder.providers
   :members: FetchedTile, TileFetcher

.. automodule:: starfish.core.experiment.builder.builder
   :members: write_experiment_json
