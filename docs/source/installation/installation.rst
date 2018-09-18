.. _installation

Installation
============

Starfish supports python 3.6 and above. To Install the starfish package, first verify that your
python version is compatible. You can check this with pip, which may be called ``pip`` or ``pip3``
depending on how you installed python.

The output should look similar to this:

.. code-block:: bash

   % pip3 --version
   pip 10.0.1 from /usr/local/lib/python3.6/site-packages/pip (python 3.6)

While not required, you may wish to set up a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_.
To do this, execute:

.. code-block:: bash

   % python -m venv .venv

Install the starfish module in edit-mode and all the dependencies for starfish:

.. code-block:: bash

   % git clone https://github.com/spacetx/starfish.git
   % pip install -e starfish

