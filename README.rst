Starfish
========

.. image:: https://travis-ci.org/spacetx/starfish.svg?branch=master
    :target: https://travis-ci.org/spacetx/starfish
    :width: 20%
.. image:: https://codecov.io/gh/spacetx/starfish/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spacetx/starfish
.. image:: https://readthedocs.org/projects/spacetx-starfish/badge/?version=latest
    :target: https://spacetx-starfish.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. ideally we could use the ..include directive here instead of copy and pasting the following
   information


.. image:: docs/source/_static/design/logo.png
    :scale: 50 %

Introduction
------------

Starfish is a Python library that lets you build scalable and modular pipelines for processing image-based transcriptomics data. Starfish is developed in the open in collaboration with SpaceTx. SpaceTx is a consortium effort to benchmark image based transcriptomic methods by applying 10 different methods on a common tissue source, standardizing the raw data formats and using standardized analysis pipelines.

For detailed information on installation and usage, see the documentation_

.. _documentation: https://spacetx-starfish.readthedocs.io/en/latest/

Quickstart
------------

We are currently in **pre-alpha**, finishing proof of concept pipelines for each of the spaceTx
contributors that leverage starfish's shared object model. Follow the links to see Jupyter notebooks showing Starfish in action. 

====================  ==========  ===================  ==================
 Assay                Loads Data  Single-FoV Pipeline  Multi-FoV Pipeline
--------------------  ----------  -------------------  ------------------
 MERFISH              [x]         [x]                  in process
 ISS                  [x]         [x]                  in process
 osmFISH              [x]         in process           [ ]
 allen_smFISH         [x]         in review            [ ]
 BaristaSeq           [x]         in process           [ ]
 DARTFISH             [x]         in review            [ ]
 ex-FISH              [x]         [ ]                  [ ]
 StarMAP              [ ]         [ ]                  [ ]
 FISSEQ               no data     no pipeline          [ ]
 seq-FISH             [ ]         [ ]                  [ ]
 Imaging Mass. Cyto.  [x]         [ ]                  [ ]
====================  ==========  ===================  ==================

Concept
-------

See this document_ for details. The diagram below describes the core pipeline components and
associated file manifests that this package plans to standardize and implement.

.. _document: https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing

.. image:: docs/source/_static/design/pipeline-diagram.png
    :alt: pipeline diagram


