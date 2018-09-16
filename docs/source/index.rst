.. Starfish documentation master file, created by
   sphinx-quickstart on Wed Sep  5 14:09:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Starfish documentation
======================

.. image:: /../../design/logo.png
    :width: 250px

The goal of *starfish* is to **prototype** a reference pipeline for the analysis of image-based 
transcriptomics data that works across technologies. This is a **work in progress** and will be 
developed in the open.

We are currently in **pre-alpha**, finishing proof of concept pipelines for each of the spaceTx 
contributors that leverage starfish's shared object model. At this time starfish is mature enough to 
support computational developers interested in adapting other assays to starfish's object model.

| assay        | loads data | single-fov pipeline complete |
| :----------- | ---------- | ---------------------------- |
| MERFISH      | [x]        | [x]                          |
| ISS          | [x]        | [x]                          |
| osmFISH      | [x]        | in process                   |
| allen_smFISH | [x]        | in review                    |
| DARTFISH     | [x]        | in review                    |
| dypFISH      | [x]        | no pipeline                  |
| ex-FISH      | [x]        | no pipeline                  |
| FISSEQ       | no data    | no pipeline                  |
| seq-FISH     | no data    | no pipeline                  |

Concept
-------

See this document_ for details. The diagram below describes the core pipeline components and 
associated file manifests that this package plans to standardize and implement.

.. _document: https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing 

.. image:: /../../design/pipeline-diagram.png
    :alt: pipeline diagram

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :hidden:

   self

.. toctree::
    installation/installation.rst

.. toctree::
    API/index.rst

.. toctree::
    contributing/contributing.md

.. toctree::
    license/license.rst

.. toctree::
    sptx-format/index.rst

.. toctree::
    usage/index.rst

.. toctree::
    glossary/glossary.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
