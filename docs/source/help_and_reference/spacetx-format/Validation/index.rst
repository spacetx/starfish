.. _schema:

.. mdinclude:: ../../../../../starfish/spacetx_format/schema/README.md

.. _cli_validate:

Validation
==========

The `starfish validate` command provides a way to check that a fileset based on the
:ref:`sptx_format` is valid. One of the schema requirements is that the codebook is version 0.0.0
and the experiment is 4.0.0 or 5.0.0.

Usage
^^^^^

starfish validate --help will provide instructions on how to use the tool:

.. program-output:: env MPLBACKEND=Agg starfish validate --help


Examples
^^^^^^^^

.. code-block:: bash

    $ starfish validate experiment tmp/experiment.json > /dev/null && echo ok

Validating the experiment, validates all of the included files. These files can also be individually validated:

.. code-block:: bash

    $ starfish validate codebook tmp/codebook.json