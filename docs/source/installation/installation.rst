.. _installation

Installation
============

Starfish supports python 3.6 and above. To install the starfish package, first verify that your
python version is compatible. You can check this by running `python --version`.

The output should look similar to this:

.. code-block:: bash

   % python --version
   Python 3.6.5

Starfish names its dependencies and lists explicit versions, due to sensitivity to subtle algorithm
changes.  For that reason, it is strongly encouraged that you set up a
virtualenv_. Create a work folder and set up the virtual environment like:

.. _virtualenv: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

.. code-block:: bash

    % mkdir starfish
    % cd starfish
    % python -m venv .venv
    % source .venv/bin/activate

Finally, the install starfish:

.. code-block:: bash

   % pip install starfish
