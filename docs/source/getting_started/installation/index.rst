.. _installation:

Installation
============

Starfish supports python 3.6 and above. To install the starfish package, first verify that your
python version is compatible. You can check this by running `python --version`.

The output should look similar to this:

.. code-block:: bash

   % python --version
   Python 3.6.5

Using virtual environments
--------------------------
   
Starfish lists minimum versions for its dependencies for access to new features and algorithms.
These more up-to-date packages may create conflicts in your existing scripts or other packages,
so you may want to set up a virtualenv_.
You can create a work folder and set up the virtual environment like:

.. _virtualenv: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

.. code-block:: bash

    % mkdir starfish
    % cd starfish
    % python -m venv .venv
    % source .venv/bin/activate

Conda_ users can set one up like so:

.. _Conda: https://www.anaconda.com/distribution/

.. code-block:: bash

    % conda create -n starfish "python>=3.6"
    % conda activate starfish

Installation for users
----------------------

Starfish can easily be installed using pip:

.. code-block:: bash

   % pip install starfish

Installation for developers
---------------------------

If you are on a mac, make sure you have the `XCode CommandLine Tools`_
installed.  Check out the code for starfish and set up a `virtual environment`_.

.. _`XCode CommandLine Tools`: https://developer.apple.com/library/archive/technotes/tn2339/_index.html
.. _`virtual environment`: #using-virtual-environments

.. code-block:: bash

    % git clone git://github.com/spacetx/starfish.git
    % cd starfish

Finally, then install starfish:

.. code-block:: bash

   % make install-dev
