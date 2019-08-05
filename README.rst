Starfish
========

.. image:: https://travis-ci.com/spacetx/starfish.svg?branch=master
    :target: https://travis-ci.com/spacetx/starfish
    :width: 20%
.. image:: https://codecov.io/gh/spacetx/starfish/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spacetx/starfish
.. image:: https://readthedocs.org/projects/spacetx-starfish/badge/?version=latest
    :target: https://spacetx-starfish.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://join-cziscience-slack.herokuapp.com/badge.svg
    :target: https://join-cziscience-slack.herokuapp.com/
    :alt: Join the CZI Science slack

.. ideally we could use the ..include directive here instead of copy and pasting the following
   information


.. image:: docs/source/_static/design/logo.png
    :scale: 50 %

Introduction
------------

Starfish is a Python library that lets you build scalable and modular pipelines for processing image-based transcriptomics data. Starfish is developed in the open in collaboration with SpaceTx. SpaceTx is a consortium effort to benchmark image based transcriptomic methods by applying 10 different methods on a common tissue source, standardizing the raw data formats and using standardized analysis pipelines.

For detailed information on installation and usage, see the documentation_.

.. _documentation: https://spacetx-starfish.readthedocs.io/en/latest/

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

- Slack: `join #starfish-users on the CZI Science Slack <https://join-cziscience-slack.herokuapp.com/>`_
- Email: `starfish@chanzuckerberg.com <mailto:starfish@chanzuckerberg.com>`_
