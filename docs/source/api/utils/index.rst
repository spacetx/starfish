.. _utils:

Utilities
=========

A number of utilities exist for simplifying work with the starfish library.

The :ref:`StarfishConfig` object can be instantiated at any point to provide configuration
regarding caching, validation, and similar low-level concerns. This is especially important
when directly calling out to IO backends like slicedimage. If a configuration setting needs
to be temporarily modified, use the :ref:`environ` context manager to set individual values.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   config.rst

.. toctree::
    logging.rst