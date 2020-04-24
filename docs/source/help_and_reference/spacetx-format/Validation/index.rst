.. _schema:

.. mdinclude:: ../../../../../starfish/spacetx_format/schema/README.md

.. _cli_validate:

Validation
==========

The `starfish validate` command provides a way to check that a fileset based on the
:ref:`sptx_format` is valid.

Usage
^^^^^

starfish validate --help will provide instructions on how to use the tool:

.. program-output:: env MPLBACKEND=Agg starfish validate --help


Examples
^^^^^^^^

::

    starfish validate experiment tmp/experiment.json > /dev/null && echo ok


Validating the experiment, validates all of the included files. These files can also be individually validated:

::

    $ starfish validate experiment tmp/codebook.json