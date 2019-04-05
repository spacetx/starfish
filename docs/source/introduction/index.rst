.. _introduction:

Introduction
============

*starfish* is a Python library which lets you build scalable pipelines for processing image-based
transcriptomics data. This is a **work in progress** and is being developed in collaboration
with users and developers of image-based transcriptomics assays.

What is *starfish*?
-------------------

**Starfish is a library for counting spots in image data**. It transforms potentially-multiplex
imaging experiments over panoramic images broken up by microscope field of view into a table of
spots (genes, proteins) localized in 3D, and can aggregate those localized spots into a cell x gene
table by comparing the physical positions of spots and cells.

To achieve this generality, starfish exposes a set of objects that allow it to work both for
discrete assays where each spot represents a molecule, and for assays that build codes across many
images. Starfish breaks up processing into fields of view that correspond to the data produced by a
microscope at a single location on a microscope slide, and is
able to process single fields of view for each of the below assays. To enable this generality
across assays, starfish requires data be converted into SpaceTx-Format, a lightweight JSON wrapper
around 2-dimensional TIFF images.

**Starfish is agnostic to the workflow runner, but does not expose a solution for processing
complete experiments -- you will need to decide how to orchestrate the processing of multiple fields
of view**.
We made this decision because our users leverage a large variety of computational infrastructures
(high performance computing
clusters, amazon web services, and google cloud) and workflow engines (snakemake, Nextflow, and
Cromwell). As a result, starfish is focusing on ensuring it is feature complete for processing
individual fields of view, exposing methods to merge data across fields of view, and has left
orchestration across fields of view to the user. Starfish runs on Mac OS X and Linux, and Windows
through the `Windows Subsystem for Linux <wsl>`_.

.. _wsl: https://docs.microsoft.com/en-us/windows/wsl/about

To validate starfish's performance, we are working in collaboration with the
:ref:`SpaceTx consortium <spacetx consortium>` to reproduce author's pipelines for each of the
following assays. This list is not comprehensive, and starfish's development team are
always excited to learn about new assays. If you want to chat, please :ref:`contact us <contact>`

====================  ==========  ===================  ==================
 Assay                Loads Data  Single-FoV Pipeline  Multi-FoV Pipeline
--------------------  ----------  -------------------  ------------------
 MERFISH              |done|      |done|               |proc|
 ISS                  |done|      |done|               |proc|
 osmFISH              |done|      |done|               |todo|
 allen_smFISH         |done|      |revw|               |todo|
 BaristaSeq           |done|      |revw|               |todo|
 DARTFISH             |done|      |revw|               |todo|
 ex-FISH              |done|      |todo|               |todo|
 StarMAP              |done|      |done|               |todo|
 seq-FISH             |done|      |proc|               |todo|
 RNAscope             |done|      |done|               |todo|
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

To dive into starfish in more detail, please proceed to the :ref:`getting started <getting started>`
section.