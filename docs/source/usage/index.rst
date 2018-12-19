.. _usage:

Starfish Usage
==============

Getting started with the CLI
----------------------------

The simplest way to get started with starfish for most users will be to try out the
command-line interface (CLI). After following the :ref:`installation <installation>`
instructions, a ``starfish`` command will be available. Running ``starfish --help``
will print out the subcommands that are available.

.. program-output:: env MPLBACKEND=Agg starfish --help

.. toctree::
   :maxdepth: 3
   :caption: CLI:

.. toctree::
   fov-builder/fov-builder.rst

.. toctree::
   validation/index.rst

.. toctree::
   configuration/index.rst

Vignettes
---------

This section provides several end-to-end usage vignettes for applying starfish to image-based
transcriptomics data. The first vignette provides an example of using starfish to format a small,
16 field of view experiment leveraging the in-situ sequencing (ISS) approach.

.. toctree::
   :maxdepth: 2
   :caption: Vignettes:

.. toctree::
   iss/iss_vignette.rst

.. toctree::
   iss/iss_cli_vignette.rst
