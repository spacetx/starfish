.. _formatting:

SpaceTx File Format
===================

Overview
--------

The SpaceTX File Format is designed to describe one or more tensors along with a codeboook to decode detected spots into targets.  The documentation for the format can be found `here`_.

.. _here: https://github.com/spacetx/starfish/tree/master/starfish/spacetx_format

Converting Data to SpaceTx Format
---------------------------------

We provide three types of tools to convert data into SpaceTx-Format. One is a `Bio-Formats`_ writer
which writes SpaceTx-Format experiments using the Bio-Formats converter. Bio-Formats can read a
variety of input formats, so might be a relatively simple approach for users familiar with those
tools.

.. _Bio-Formats: https://www.openmicroscopy.org/bio-formats/

Second, we provide a mechanism by which the user organizes the data as 2D tiles with a clearly defined filename schema, and a conversion tool.  There is :ref:`documentation and an example <format_structured_data>` for that mechanism.

If neither of these models fit, then we provide a :ref:`generalized mechanism <advanced_formatting>` where conversion is managed through a set of interfaces where the user provides python code responsible for obtaining the data corresponding to each 2D tile.  :ref:`Example formatters <data_conversion_examples>` for a variety of datasets are also available.  This same interface can also be used to :ref:`directly load data <tilefetcher_loader>`, although there may be performance implications in doing so.
