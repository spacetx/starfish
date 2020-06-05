.. _installation:

Installation
============

Starfish supports python 3.6 and above (python 3.7 recommended). To install the starfish package,
first verify that your python version is compatible. You can check this by running :code:`python
--version`.

The output should look similar to this:

.. code-block:: bash

   $ python --version
   Python 3.7.7

.. warning::
    While starfish itself has no known issues with python 3.8, scikit-image is not fully
    compatible with python 3.8. As such, installation of scikit-image, as part of starfish
    installation, may unexpectedly fail. The workaround is to install numpy first before
    installing starfish or scikit-image.


Using virtual environments
--------------------------

Starfish lists minimum versions for its dependencies for access to new features and algorithms.
These more up-to-date packages may create conflicts in your existing scripts or other packages,
so we recommend using a virtualenv_. You can create a work folder and set up the virtual
environment like:

.. _virtualenv: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

.. code-block:: bash

    $ mkdir starfish
    $ cd starfish
    $ python -m venv .venv
    $ source .venv/bin/activate

Conda_ users can set one up like so:

.. _Conda: https://www.anaconda.com/distribution/

.. code-block:: bash

    $ conda create -n starfish "python=3.7"
    $ conda activate starfish

Installing *starfish*
---------------------

Starfish can easily be installed using pip:

.. code-block:: bash

    $ pip install starfish

.. note::
    If using python 3.8, first install numpy using pip before installing starfish.

To use napari for interactive image visualization via :py:func:`.display` you must also
install napari:

.. code-block:: bash

    $ pip install starfish[napari]

Interactive visualization with napari also requires using Qt (e.g. by running the magic command
`%gui qt` in a jupyter notebook or ipython shell.)

Installing *starfish* on Windows
--------------------------------

Windows users can install starfish in the same way. Again, we recommend using a conda or virtual
environment with python 3.7. Here is how you would install starfish in a virtual environment
created with python's ``venv`` module:

.. code-block:: bat

    > mkdir starfish
    > cd starfish
    > python -m venv .venv
    > .venv\Scripts\activate.bat
    > pip install starfish
    > pip install starfish[napari]

.. note::
    Python 3.8 has trouble installing scikit-image v0.15.0 and the ``pip install numpy``
    workaround does not solve this issue on Windows.

Jupyter notebook
----------------

To run starfish in a jupyter notebook (recommended for creating an image processing pipeline) add
the virtualenv kernel to jupyter by activating your virtual environment and then:

.. code-block:: bash

    $ python -m ipykernel install --user --name=<venv_name>

Now you should be able to select ``venv_name`` as the kernel in a jupyter notebook to have access
to the starfish library.
