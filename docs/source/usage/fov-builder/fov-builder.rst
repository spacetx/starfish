.. _cli_build:

Synthetic experiments
=====================

Building synthetic SpaceTx-specification compliant experiments
--------------------------------------------------------------

starfish provides a tool to construct example datasets that can be used to test software for use with our formats.
This tool generates spaceTx-specification experiments with tunable sizes and shapes, but the images are randomly generated and do not contain biologically meaningful data.

Usage
-----

starfish build --help will provide instructions on how to use the tool:

.. program-output:: env MPLBACKEND=Agg starfish build --help

Examples
--------

Build a 3-field of view experiment with 2 channels and 8 hybridization rounds per primary image stack that samples z 30 times.
The experiment has both a dots image and a nuclei image, but these have only one channel and round each.
The size of the (x,y) tiles cannot be modified at this time.

::

    mkdir tmp
    OUTPUT_DIR=tmp
    starfish build \
        --fov-count 3 \
        --primary-image-dimensions '{"r": 8, "c": 2, "z": 30}' \
        --dots-dimensions '{"r": 1, "c": 1, "z": 30}' \
        --nuclei-dimensions '{"r": 1, "c": 1, "z": 30}' \
        ${OUTPUT_DIR}
