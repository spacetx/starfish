.. _spacetx consortium:

SpaceTx Consortium
==================

The SpaceTx consortium is a group led by Ed Lein and the Allen Institute for Brain Science that
is engaged in benchmarking image-based transcriptomics assays. In collaboration with the lead
developers of 10+ image-based transcriptomics assays, they are applying each technology to adjacent
sections of mouse and human primary visual cortex. The goal of this experiment is to understand
what assay(s) are best applied to (a) spatially localizing cells by type based on marker genes and
(b) obtaining robust phenotypes of cells by measuring RNA from thousands of genes.

Consortium members include:

- Ed Boyden, MIT
- Long Cai, Caltech
- Fei Chen, Broad Institute
- Karl Dieserroth, Stanford
- Ed Lein, Allen Institute for Brain Science
- Sten Linnarrson, Karolinska Institutet
- Joakim Lundeberg, SciLifeLab
- Jeffrey Moffit, Harvard
- Mats Nilsson, SciLifeLab
- Aviv Regev, Broad Institute
- Anthony Zador, Cold Spring Harbor Laboratory
- Kun Zhang, University of California San Diego
- Xiaowei Zhuang, Harvard

Pipelines
---------

Follow the links in the table below to see pipelines from each of the SpaceTx groups ported to starfish.
All of the example pipelines can be found in the `notebooks directory <https://github.com/spacetx/starfish/tree/master/notebooks/>`_.

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
