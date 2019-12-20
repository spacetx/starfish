"""
.. _format_structured_data:

Format Structured Data
======================

Format a dataset where the filenames of the tiles provide most of the metadata required to
organize them into the 5D set (image_type, fov, round, ch, zplane) of 2D tiles  (y, x).  The
remaining metadata, namely physical coordinates of the tiles, are provided in a CSV file.

Image types generally at least includes 'primary' (defined in the code by
:py:attr:`~starfish.experiment.experiment.FieldOFView.PRIMARY_IMAGES`).  It may also be 'nuclei' or
'dots'.

The stem of the tiles' filenames must be
`<image_type>-f<fov_id>-r<round_label>-c<ch_label>-z<zplane_label>`.  The extension should be one
of the supported tile formats (:py:class:`~slicedimage.ImageFormat`).  For example, the file
`nuclei-f0-r2-c3-z33.tiff` would belong to the nuclei image, fov 0, round 2, channel 3, zplane 33.

The CSV file must have a header, and must contain the following columns:

====== ======== ====================================================================================
Column Required Notes
------ -------- ------------------------------------------------------------------------------------
fov    yes
round  yes
ch     yes
zplane yes
xc_min yes      This should be the minimum value of the x coordinate of the tile.
xc_max yes      This should be the maximum value of the x coordinate of the tile.
yc_min yes      This should be the minimum value of the y coordinate of the tile.
yc_max yes      This should be the maximum value of the y coordinate of the tile.
zc_min no       This should be the minimum value of the z coordinate of the tile.
zc_max no       This should be the maximum value of the z coordinate of the tile.
====== ======== ====================================================================================
"""

###################################################################################################
# Create some synthetic data to form into a trivial experiment.
# -------------------------------------------------------------
# fov0 contains r0, ch0, zplane0-2.  All tiles share the same physical coordinates.
# fov1 contains r0-1, ch0, zplane0-2.  All tiles share the same physical coordinates.
import csv
import os
import numpy as np
import skimage.io
import tempfile

# columns: r, ch, zplane
fovs = [
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
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
        'zc_max': 0.005,
    },
    {
        'xc_min': 0.1,
        'xc_max': 0.2,
        'yc_min': 0.0,
        'yc_max': 0.1,
        'zc_min': 0.005,
        'zc_max': 0.005,
    },
]

data = np.zeros((10, 10), dtype=np.float32)

# create example image tiles that adhere to the file naming schema
inputdir = tempfile.TemporaryDirectory()
for fov_id, fov in enumerate(fovs):
    for round_label, ch_label, zplane_label in fov:
        path = os.path.join(
            inputdir.name, f"primary-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff")
        skimage.io.imsave(path, data)

# write coordinates file
coordinates_path = os.path.join(inputdir.name, "coordinates.csv")
with open(coordinates_path, "w") as fh:
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

###################################################################################################
# Directory contents
# ------------------

for file in sorted(os.listdir(inputdir.name)):
    print(file)

###################################################################################################
# Contents of coordinates.csv
# ---------------------------

with open(coordinates_path, "r") as fh:
    print(fh.read())

###################################################################################################
# Construct an experiment out of the raw files.
# ---------------------------------------------

outputdir = tempfile.TemporaryDirectory()
from slicedimage import ImageFormat
from starfish.core.experiment.builder.structured_formatter import format_structured_dataset

format_structured_dataset(
    inputdir.name,
    coordinates_path,
    outputdir.name,
    ImageFormat.TIFF,
)

###################################################################################################
# List the output directory
# -------------------------

for file in sorted(os.listdir(outputdir.name)):
    print(file)

###################################################################################################
# Load up the experiment
# ----------------------
from starfish import Experiment

exp = Experiment.from_json(os.path.join(outputdir.name, "experiment.json"))
print(exp.fovs())
print(repr(exp.fov().get_image('primary')))
