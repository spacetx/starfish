.. _installation:

Installation
============

Starfish supports python 3.9-3.12. To install the starfish package,
first verify that your python version is compatible. You can check this by running :code:`python
--version`.

The output should look similar to this:

.. code-block:: bash

   $ python --version
   Python 3.9.18

.. warning::
    While starfish itself should work on any operating system, some napari dependencies might not be
    compatible with Apple Silicon or Windows. As such, installation of napari, as part of starfish[napari]
    installation, may unexpectedly fail. The workaround is to install napari first before
    installing starfish or updating the dependencies via cloning the project and working in development mode.


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

    $ conda create -n starfish "python=3.9"
    $ conda activate starfish

Installing *starfish*
---------------------

Starfish can easily be installed using pip:

.. code-block:: bash

    $ pip install starfish

for the most updated version install directly from Github (starfish release on PyPI might be a few months behind the repo's master branch):

.. code-block:: bash

    $ pip install starfish@git+https://github.com/spacetx/starfish.git

To use napari for interactive image visualization via :py:func:`.display` you must also
install napari:

.. code-block:: bash

    $ pip install starfish[napari]

.. note::
    If using Windows or Apple Silicon (M1+), one might need to first install napari using pip before installing starfish (see below). Also, interactive visualization with napari requires using Qt (for more information about Qt backend see choosing-a-different-qt-backend_).

.. _choosing-a-different-qt-backend: https://napari.org/dev/tutorials/fundamentals/installation.html#choosing-a-different-qt-backend

To install starfish with both napari and jupyter for notebook support:

.. code-block:: bash

    $ pip install starfish[jupyter]

Installing *starfish* on Windows
--------------------------------

Windows (cmd.exe) users can install starfish in the same way. Again, we recommend using a conda or virtual
environment with python 3.9+. Here is how you would install starfish in a virtual environment
created with python's ``venv`` module:

.. code-block:: bat

    > mkdir starfish
    > cd starfish
    > python -m venv .venv
    > .venv\Scripts\activate.bat
    > pip install napari[all]
    > pip install starfish

.. note::
    If you encounter issues, you need to update the dependencies via cloning the project and working in development mode.

Jupyter notebook
----------------

To run starfish in a jupyter notebook (recommended for creating an image processing pipeline) add
the virtualenv kernel to jupyter by activating your virtual environment and then:

.. code-block:: bash

    $ python -m pip install jupyter
    $ python -m ipykernel install --user --name=<venv_name>

Now you should be able to select ``venv_name`` as the kernel in a jupyter notebook to have access
to the starfish library.

Installing *starfish* in development mode
-----------------------------------------

If you need to resolve dependency issues with napari and jupyter or want to tinker with the
starfish package, it is best to work in development mode. 
If you are on a mac, make sure you have the `XCode CommandLine Tools`_
installed.

.. _`XCode CommandLine Tools`: https://developer.apple.com/library/archive/technotes/tn2339/_index.html

Check out the code for starfish:

.. code-block:: bash

    $ git clone https://github.com/spacetx/starfish.git
    $ cd starfish

Set up a `virtual environment`_:

.. _`virtual environment`: #using-virtual-environments

.. code-block:: bash

    $ python -m venv .venv
    $ source .venv/bin/activate

Install starfish:

.. code-block:: bash

    $ make install-dev

Update dependencies for napari and jupyter:

.. code-block:: bash

    $ make -B requirements/REQUIREMENTS-NAPARI-CI.txt
    $ make -B requirements/REQUIREMENTS-JUPYTER.txt

Install napari and jupyter:

.. code-block:: bash

    $ pip install -r requirements/REQUIREMENTS-NAPARI-CI.txt
    $ pip install -r requirements/REQUIREMENTS-JUPYTER.txt
