.. _validation:

Validation
==========

Validators are provided for validating a SpaceTx fileset against the :ref:`schema`.

Validators
----------

.. autoclass:: starfish.core.spacetx_format.util.SpaceTxValidator
   :members:
   :exclude-members: fuzz_object

Helpers
-------

In addition, the starfish.spacetx_format.validate_sptx module contains helpers to simplify
iterating over the tree of json files and their respective schemas.


.. automodule:: starfish.core.spacetx_format.validate_sptx
   :members:


Error messages
--------------

Descriptive error messages are printed as warnings while validation takes place.
For example:

::

    starfish/starfish/core/spacetx_format/util.py:82: UserWarning:
     'contents' is a required property
            Schema:                 https://github.com/spacetx/starfish/starfish/file-format/schema/fov-manifest.json
            Subschema level:        0
            Path to error:          required
            Filename:               ../field_of_view/field_of_view.json
    
      warnings.warn(message)


This message tells you which schema has failed validation (``fov-manifest.json``), what type of error
has been encountered (``a required field is missing``), and the name of the file which is invalid
(``field_of_view.json``) if it has been provided. Validation of a json object will simply omit the
``Filename:`` field.
