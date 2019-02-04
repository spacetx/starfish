.. _introduction:

Introduction
============

*starfish* is a Python library which lets you build scalable pipelines for processing image-based
transcriptomics data. This is a **work in progress** and will be developed in the open.

Assay support
~~~~~~~~~~~~~

We are currently developing proof-of-concept pipelines for each of image-based
transcriptomics assays that are being developed by the SpaceTx groups.

====================  ==========  ===================  ==================
 Assay                Loads Data  Single-FoV Pipeline  Multi-FoV Pipeline
--------------------  ----------  -------------------  ------------------
 MERFISH              |done|      |done|               |proc|
 ISS                  |done|      |done|               |proc|
 osmFISH              |done|      |proc|               |todo|
 allen_smFISH         |done|      |revw|               |todo|
 BaristaSeq           |done|      |proc|               |todo|
 DARTFISH             |done|      |revw|               |todo|
 ex-FISH              |done|      |todo|               |todo|
 StarMAP              |todo|      |todo|               |todo|
 FISSEQ               |todo|      |todo|               |todo|
 seq-FISH             |todo|      |todo|               |todo|
 Imaging Mass. Cyto.  |done|      |todo|               |todo|
====================  ==========  ===================  ==================

Legend:

- |done| - Done
- |revw| - In Review
- |proc| - In Process
- |todo| - TODO
- |none| - Not supported

.. |done| unicode:: U+2705 .. White Heavy Check Mark
.. |proc| unicode:: U+1F51C .. Soon Arrow
.. |revw| unicode:: U+1F91E .. Crossed Fingers
.. |todo| unicode:: U+1F532 .. Black Square Button
.. |none| unicode:: U+274C .. Cross Mark

Concept
~~~~~~~

See this document_ for details. The diagram below describes the core pipeline components and
associated file manifests that this package plans to standardize and implement.

.. _document: https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing

.. image:: /_static/design/pipeline-diagram.png
    :alt: pipeline diagram
