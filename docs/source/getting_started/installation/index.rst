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

    % git clone git://github.com/spacetx/starfish.git
    % cd starfish
    % python -m venv .venv
    % source .venv/bin/activate

Finally, then install starfish:

.. code-block:: bash

   % make install-dev

Step by Step Installation on Windows
--------------------------------------

We recommend using starfish with a Windows Linux Subsystem (WSL)

Instructions on how to download and install a new WSL can be found here: `WSL install manual`_

.. _`WSL install manual`: https://docs.microsoft.com/en-us/windows/wsl/install-manual

Once your WSL is running, run an apt-get update and install pip

.. code-block:: bash

    % sudo apt-get update
    % sudo apt-get install python3-pip

Clone the starfish repo

.. code-block:: bash

    % git checkout git@github.com:spacetx/starfish.git
    % cd starfish

Install create and activate a virtualenv

.. code-block:: bash

    % python3 -m pip install --user virtualenv
    % python3 -m virtualenv venv
    % source venv/bin/activate

Finally, then install starfish:

.. code-block:: bash

   % make install-dev
