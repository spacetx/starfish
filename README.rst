Starfish
========

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fstarfish.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC
    :target: https://forum.image.sc/tag/starfish
    :alt: Image.sc forum

.. image:: https://img.shields.io/pypi/v/starfish   
    :target: https://pypi.org/project/starfish/
    :alt: PyPI
    
.. image:: https://img.shields.io/pypi/dm/starfish   
   :target: https://pypistats.org/packages/starfish
   :alt: PyPI - Downloads

.. image:: https://readthedocs.org/projects/spacetx-starfish/badge/?version=latest
    :target: https://spacetx-starfish.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://travis-ci.com/spacetx/starfish.svg?branch=master
    :target: https://travis-ci.com/spacetx/starfish
    
.. image:: https://codecov.io/gh/spacetx/starfish/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spacetx/starfish
    

.. ideally we could use the ..include directive here instead of copy and pasting the following
   information


.. image:: docs/source/_static/design/logo.png
    :scale: 50 %

Introduction
------------

Starfish is a Python library that lets you build scalable and modular pipelines for processing image-based transcriptomics data. Starfish is developed in the open in collaboration with SpaceTx. SpaceTx is a consortium effort to benchmark image based transcriptomic methods by applying 10 different methods on a common tissue source, standardizing the raw data formats and using standardized analysis pipelines.

For detailed information on installation and usage, see the documentation_.

.. _documentation: https://spacetx-starfish.readthedocs.io/en/latest/

To see what improvements the developers have planned for starfish, please see the :ref:`roadmap`.

Quickstart
------------

We are currently in **beta**, finishing proof of concept pipelines for each of the spaceTx
contributors that leverage starfish's shared object model. Follow the links in the table below
to see starfish in action on particular assay types. Or, browse our our notebooks directory `here <https://github.com/spacetx/starfish/tree/master/notebooks/>`_.

====================  ==========  ===================  ==================
 Assay                Loads Data  Single-FoV Pipeline  Multi-FoV Pipeline
--------------------  ----------  -------------------  ------------------
 MERFISH              [x]         [x] mer_             in process
 ISS                  [x]         [x] iss_             [x]
 osmFISH              [x]         [x] osm_             [ ]
 smFISH               [x]         [x] 3ds_             [x]
 BaristaSeq           [x]         [x] bar_             [ ]
 DARTFISH             [x]         [x] dar_             [ ]
 ex-FISH              [x]         [ ]                  [ ]
 StarMAP              [x]         [x] str_             [ ]
 seq-FISH             [x]         in process: seq_     [ ]
 FISSEQ               no data     no pipeline          [ ]
====================  ==========  ===================  ==================

.. _mer: https://github.com/spacetx/starfish/blob/master/notebooks/MERFISH.ipynb
.. _iss: https://github.com/spacetx/starfish/blob/master/notebooks/ISS.ipynb
.. _osm: https://github.com/spacetx/starfish/blob/master/notebooks/osmFISH.ipynb
.. _bar: https://github.com/spacetx/starfish/blob/master/notebooks/BaristaSeq.ipynb
.. _dar: https://github.com/spacetx/starfish/blob/master/notebooks/DARTFISH.ipynb
.. _str: https://github.com/spacetx/starfish/blob/master/notebooks/STARmap.ipynb
.. _seq: https://github.com/spacetx/starfish/blob/master/notebooks/SeqFISH.ipynb
.. _3ds: https://github.com/spacetx/starfish/blob/master/notebooks/smFISH.ipynb

Contributing
------------

We are very interested in contributions! See our contributing.rst_ and documentation_ for more information.

.. _documentation: https://spacetx-starfish.readthedocs.io/en/latest/
.. _contributing.rst: https://github.com/spacetx/starfish/blob/master/CONTRIBUTING.rst

Contact Starfish
----------------

- Forum: ask on the `Image.sc forum <https://forum.image.sc/tag/starfish>`_
- Email: `starfish@chanzuckerberg.com <mailto:starfish@chanzuckerberg.com>`_
