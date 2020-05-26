"""
.. _format_structured_data:

Format Structured Data
======================

The starfish package contains convenient functions for writing SpaceTx formatted experiments in
the :py:mod:`starfish.experiment.builder` module. Converting experiment data into SpaceTx format
with :py:func:`.format_structured_dataset` is best
for a small number of 2D images or if you have a preferred scripting language that isn't Python.
Using your tool of choice, prepare your data as a "structured dataset".

A structured dataset is a collection of 2D tiles where the filenames provide most of the
experiment metadata required to organize them into the 5D set (image_type, fov, round, ch,
zplane) of 2D tiles (y, x). The remaining metadata, namely physical coordinates of the tiles,
are provided in a CSV file.

The image_types in an experiment generally at least include 'primary' images, which are the
images containing single-molecule FISH, barcoded molecules, or proteomics data. An experiment can
also include 'nuclei' or 'dots', which is a special image that contains all the molecules in a
barcoded experiment. You can name your image_types anything you'd like, such as 'dapi', and
`starfish validate` will throw a warning if it does not recognize the image_type but it should
be compatible with the rest of the starfish package.

The stem of the tiles' filenames must be
`<image_type>-f<fov_id>-r<round_label>-c<ch_label>-z<zplane_label>`. The extension should be one
of the formats supported by :py:class:`slicedimage.ImageFormat` (tiff, png, npy). For example,
the file `nuclei-f0-r2-c3-z33.tiff` would belong to the nuclei image, fov 0, round 2, channel 3,
zplane 33.

The physical coordinates CSV file must have a header, and must contain the following columns:

.. table::
   :widths: 25 25 50
   :class: "table-bordered"

   +--------+----------+-------------------------------------------------------------------+
   | Column | Required | Notes                                                             |
   +========+==========+===================================================================+
   | fov    | yes      |                                                                   |
   +--------+----------+-------------------------------------------------------------------+
   | round  | yes      |                                                                   |
   +--------+----------+-------------------------------------------------------------------+
   | ch     | yes      |                                                                   |
   +--------+----------+-------------------------------------------------------------------+
   | zplane | yes      |                                                                   |
   +--------+----------+-------------------------------------------------------------------+
   | xc_min | yes      | This should be the minimum value of the x coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+
   | xc_max | yes      | This should be the maximum value of the x coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+
   | yc_min | yes      | This should be the minimum value of the y coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+
   | yc_max | yes      | This should be the maximum value of the y coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+
   | zc_min | no       | This should be the minimum value of the z coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+
   | zc_max | no       | This should be the maximum value of the z coordinate of the tile. |
   +--------+----------+-------------------------------------------------------------------+

Because each image_type is treated as a separate set of images, you need a different
coordinates CSV file for each image_type. Therefore, each image_type must be converted in its own
directory containing all the 2D image files and and CSV file.
For example, even if the nuclei images were acquired in every round along with the primary
images, they should be organized into two directories, each starting at round 0 and channel 0.
After converting with :py:func:`.format_structured_dataset`, the output files can be combined into a
single directory and the `experiment.json` file should be updated to include all the data
manifests (see :ref:`sptx_format` for reference).

To illustrate the overall process, we will walk through a dummy example below.
"""

###################################################################################################
# Create some synthetic data to form into a trivial experiment.
# -------------------------------------------------------------
# This dummy data represents an experiment that acquired primary and nuclei images
# simultaneously, with one channel for primary FISH spots and one channel for stained nuclei.
# There are two FOVs , 2 rounds and 3 z-plane. All tiles share the same physical coordinates.

import csv
import os
import numpy as np
import shutil
import skimage.io
import tempfile

# columns: r, ch, zplane
fovs = [
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ],
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ],
]

coordinates_of_fovs = [
    {
        'xc_min': 0.0,
        'xc_max': 0.1,
        'yc_min': 0.0,
        'yc_max': 0.1,
        'zc_min': 0.005,
        'zc_max': 0.010,
    },
    {
        'xc_min': 0.1,
        'xc_max': 0.2,
        'yc_min': 0.0,
        'yc_max': 0.1,
        'zc_min': 0.005,
        'zc_max': 0.010,
    },
]

data = np.zeros((10, 10), dtype=np.float32)

# create example image tiles that adhere to the structured data schema
inputdir = tempfile.TemporaryDirectory()
primary_dir = os.path.join(inputdir.name, "primary_dir")
nuclei_dir = os.path.join(inputdir.name, "nuclei_dir")
os.mkdir(primary_dir)
os.mkdir(nuclei_dir)

for fov_id, fov in enumerate(fovs):
    for round_label, ch_label, zplane_label in fov:
        primary_path = os.path.join(
            primary_dir, f"primary-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff")
        nuclei_path = os.path.join(
            nuclei_dir, f"nuclei-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff")
        skimage.io.imsave(primary_path, data)
        skimage.io.imsave(nuclei_path, data)

# write coordinates file for primary and nuclei in their respective directories
with open(os.path.join(primary_dir, "coordinates.csv"), "w") as fh:
    csv_writer = csv.DictWriter(
        fh,
        [
            'fov', 'round', 'ch', 'zplane',
            'xc_min', 'yc_min', 'zc_min', 'xc_max', 'yc_max', 'zc_max',
        ]
    )
    csv_writer.writeheader()
    for fov_id, (fov_info, coordinate_of_fov) in enumerate(zip(fovs, coordinates_of_fovs)):
        for round_label, ch_label, zplane_label in fov:
            tile_coordinates = coordinate_of_fov.copy()
            tile_coordinates.update({
                'fov': fov_id,
                'round': round_label,
                'ch': ch_label,
                'zplane': zplane_label,
            })
            csv_writer.writerow(tile_coordinates)

# copy same coordinates file to nuclei directory
shutil.copyfile(
    os.path.join(primary_dir, "coordinates.csv"), os.path.join(nuclei_dir, "coordinates.csv"))

###################################################################################################
# Directory contents
# ------------------

for dir in [primary_dir, nuclei_dir]:
    print("\n")
    for file in sorted(os.listdir(dir)):
        print(file)

###################################################################################################
# Contents of coordinates.csv
# ---------------------------

# just printing one of the coordinates.csv since they are identical
with open(os.path.join(primary_dir, "coordinates.csv"), "r") as fh:
    print(fh.read())

###################################################################################################
# Convert structured data into SpaceTx Format
# -------------------------------------------
# The primary and nuclei directories must be converted separately.

outputdir = tempfile.TemporaryDirectory()
primary_out = os.path.join(outputdir.name, "primary")
nuclei_out = os.path.join(outputdir.name, "nuclei")
os.makedirs(primary_out, exist_ok=True)
os.makedirs(nuclei_out, exist_ok=True)

from slicedimage import ImageFormat
from starfish.experiment.builder import format_structured_dataset

format_structured_dataset(
    primary_dir,
    os.path.join(primary_dir, "coordinates.csv"),
    primary_out,
    ImageFormat.TIFF,
)
format_structured_dataset(
    nuclei_dir,
    os.path.join(nuclei_dir, "coordinates.csv"),
    nuclei_out,
    ImageFormat.TIFF,
)

###################################################################################################
# Output directory contents
# -------------------------

for dir in [primary_out, nuclei_out]:
    print("\n")
    for file in sorted(os.listdir(dir)):
        print(file)

###################################################################################################
# Merge outputs by modifying experiment.json
# ------------------------------------------
# Each directory contains an experiment.json and a codebook.json. We'll use the ones in primary
# and the redundant JSONs can be safely deleted. The experiment.json needs to be modified to include
# the nuclei.json manifest by adding a single line.

with open(os.path.join(primary_out, "experiment.json"), "r+") as fh:
    contents = fh.readlines()
    print("original experiment.json\n")
    print("".join(contents))
    contents[3] = ",".join([contents[3].strip("\n"),"\n"])
    contents.insert(4, '\t"nuclei": "../nuclei/nuclei.json"\n')  # new_string should end in a newline
    fh.seek(0)  # readlines consumes the iterator, so we need to start over
    fh.writelines(contents)  # No need to truncate as we are increasing filesize
    fh.seek(0)
    print("\nmodified experiment.json\n")
    print(fh.read())

###################################################################################################
# Don't forget to replace the fake codebook.json
# ----------------------------------------------
# There are no starfish tools for creating a codebook. You can write the JSON manually or write a
# script to do it for you. Be sure the format matches the examples in
# :ref:`SpaceTx Format<sptx_codebook_format>`.

# this is the placeholder codebook.json
with open(os.path.join(primary_out, "codebook.json"), "r") as fh:
    print(fh.read())

###################################################################################################
# Load up the experiment
# ----------------------
from starfish import Experiment

exp = Experiment.from_json(os.path.join(primary_out, "experiment.json"))
print(exp.fovs())
