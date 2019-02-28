.. _installation:

Installation
============

Starfish supports python 3.6 and above. To install the starfish package, first verify that your
python version is compatible. You can check this by running `python --version`.

The output should look similar to this:

.. code-block:: bash

   % python --version
   Python 3.6.5

Installation for users
----------------------

Starfish names its dependencies and lists explicit versions, due to sensitivity to subtle algorithm
changes.  For that reason, it is strongly encouraged that you set up a
virtualenv_. Create a work folder and set up the virtual environment like:

.. _virtualenv: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

.. code-block:: bash

    % mkdir starfish
    % cd starfish
    % python -m venv .venv
    % source .venv/bin/activate

Finally, then install starfish:

.. code-block:: bash

   % pip install starfish

Installation for developers
---------------------------

If you are on a mac, make sure you have the `XCode CommandLine Tools`_
installed.  Check out the code for starfish and set up a virtualenv_.

.. _`XCode CommandLine Tools`: https://developer.apple.com/library/archive/technotes/tn2339/_index.html
.. _virtualenv: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

.. code-block:: bash

    % git checkout git@github.com:spacetx/starfish.git
    % cd starfish
    % python -m venv .venv
    % source .venv/bin/activate

Finally, then install starfish:

.. code-block:: bash

   % make install-dev
