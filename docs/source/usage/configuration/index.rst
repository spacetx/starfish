.. _cli_config:

Configuration
=============

`starfish` commands can all be configured in a unified way.

If the :ref:`STARFISH_CONFIG <env_config>` environment variable is set, then it will
be loaded, either (1) as a JSON file if the value starts with ``@`` or (2) as a JSON
string for all other values. For example:

::

    export STARFISH_CONFIG=@~/.my.starfish

::

    export STARFISH_CONFIG='{"verbose": false}'

Otherwise, ``~/.starfish/config`` will be loaded if it exists and is a valid JSON file.
If neither is true, then only the :ref:`default values <env_defaults>` will apply.


Additionally, the individual properties from the configuration JSON can be set by
environment variable. These values take precedence if also set in the configuration
file. For example:

::

 {
     "validation": {
         "strict": true
     }
 }

can also be specified as:

::

    export STARFISH_VALIDATION_STRICT=true

Other valid values for "true" are: "TRUE", "True", "yes", "y", "1", "on", and "enabled".

.. _env_defaults:

Default values
--------------

To not require any mandatory configuration, the following values will be
assumed if no configuration is available.

::

 {
     "slicedimage": {
         "caching": {
             "debug": false,
             "directory": "~/.starfish/cache",
             "size_limit": 5e9
         },
     },
     "validation": {
         "strict": false
     },
     "verbose": true
 }

.. _env_main:

Main environment variables
--------------------------

.. _env_config:

``STARFISH_CONFIG``
~~~~~~~~~~~~~~~~~~~

The primary configuration variable is ``STARFISH_CONFIG``.
By default, it is equivalent to ``~/.starfish/config`` which need not exist.

.. _env_validate_strict:

``STARFISH_VALIDATION_STRICT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strict validation will run the equivalent of :ref:`starfish validate <cli_validate>`
on every experiment that is loaded. Execution will halt with an exception. The initial
loading of an experiment will also take longer since files must be downloaded for
validation. If caching is enabled, the overall impact should not be significant.

.. _env_verboase:

``STARFISH_VERBOSE``
~~~~~~~~~~~~~~~~~~~~

Whether or not various commands should should print internal status messages.
By default, true.

.. _env_backend:

Backend environment variables
-----------------------------

Starfish currently uses the slicedimage library as a backend for storing large image sets.
Configuration values starting with ``SLICEDIMAGE_`` (or optionally, ``STARFISH_SLICEDIMAGE_``)
will be passed to the backend without modification.

.. _env_slicedimage_caching_size_limit:

``SLICEDIMAGE_CACHING_SIZE_LIMIT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum size of the :ref:`cache directory <env_slicedimage_caching_directory>`.
By default, size_limit is 5GB. Setting size_limit to 0 disables caching.

.. _env_slicedimage_caching_directory:

``SLICEDIMAGE_CACHING_DIRECTORY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directory where the cache will be stored.
By default, the directory ``~/.starfish/cache``.

.. _env_slicedimage_debug:

``SLICEDIMAGE_CACHING_DEBUG``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether or not to print which files are being cached.
By default, false.
