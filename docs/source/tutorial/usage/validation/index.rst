.. _cli_validate:

Validation
==========

The `starfish validate` command provides a way to check that a fileset based on the
:ref:`sptx_format` is valid.

Usage
-----

starfish validate --help will provide instructions on how to use the tool:

.. program-output:: env MPLBACKEND=Agg starfish validate --help


Examples
--------

As a simple example, we can show that the :ref:`synthetic experiment <cli_build>`
created in the previous section is valid:

::

    starfish validate --experiment-json tmp/experiment.json > /dev/null && echo ok


Building a :ref:`synthetic experiment <cli_build>` can provide you with a template that
you can use to model your own data. If you then modify that experiment incorrectly, you
might see the following validation warning:

::

    $ starfish validate --experiment-json tmp/experiment.json
    
             _              __ _     _
            | |            / _(_)   | |
         ___| |_ __ _ _ __| |_ _ ___| |__
        / __| __/ _` | '__|  _| / __| '_  `
        \__ \ || (_| | |  | | | \__ \ | | |
        |___/\__\__,_|_|  |_| |_|___/_| |_|
    
    
    /scratch/repos/starfish/sptx_format/util.py:82: UserWarning:
     Additional properties are not allowed ('nuclei_dimensions', 'dots_dimensions' were unexpected)
            Schema:                 unknown
            Subschema level:        0
            Path to error:          properties/images/additionalProperties
            Filename:               experiment.json
    
      warnings.warn(message)
    /scratch/repos/starfish/sptx_format/util.py:82: UserWarning:
     'codeword' is a required property
            Schema:                 unknown
            Subschema level:        0
            Path to error:          properties/mappings/items/required
            Filename:               codebook.json
    
      warnings.warn(message)
