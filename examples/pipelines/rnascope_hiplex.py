"""
.. _RNAScope_processing_example:

Single Field of View for RNAScope HiPlex
=========================================

ACDBio has a 12-gene HiPlex assay that uses 3 rounds of fluorescence imaging with 4 fluorescence RNA
channels and 1 DAPI-stained nuclei channel. In the example data provided by ACDBio, these 15 images
are saved as single-plane RGB tiffs with filenames that indicate the round and channel. A key for
mapping images to genes is included in list.txt. This vignette will demonstrate how to format the
data into SpaceTx format, register the images into the same coordinate system, and process them
into cell x gene expression matrices using modules contained in starfish.

Feedback from ACDBio:

* Blob detection needs improvement in:

  * Low-contrast regions - skip background subtraction and enhance contrast instead

  * Clustered regions - use pre-defined spot size to break up clusters and count as single spots

  * High-intensity saturated regions - use pre-defined spot size and intensity to resolve saturated regions as single spots

  * Airy disks - either remove airy disks prior to spot finding or modify spot finding algorithm to account for airy disk pattern

* Image registration could be improved with rotation (low priority)

* Cell Segmentation (watershed)

  * sum intensities instead of max

  * connect nuclear fragments with morphological operation

* Expression Matrix

  * Summary of data like dot size, intensity, etc - this is in the DecodedIntensityTable object

  * Further analysis e.g. normalization - not in the scope of starfish

* Show blob results overlaid on raw

* overlay_spot_calls() wrong coordinates?
"""

###################################################################################################
# Convert RGB images to greyscale images
# --------------------------------------
# starfish and the SpaceTx format expect greyscale images so the RGB data must first be converted.
# This can be done in many tools, such as MATLAB and ImageJ. Here we chose to use Python Imaging
# Library (PIL):

import os
import numpy as np
from shutil import copy2
from PIL import Image


def convert_rgb_to_greyscale(input_dir, output_dir):
    """
    Original example data is stored in a directory named 'original_data' and consists of 15 RGB
    images and 1 text file.
    Copy everything into a new `output_dir` and convert any RGB tiffs to greyscale.

    └── original_data
            ├── list.txt
            ├── R1_DAPI.tif
            ├── R1_Probe T1.tif
            ├── R1_Probe T2.tif
            ├── R1_Probe T3.tif
            ├── R1_Probe T4.tif
            ├── R2_DAPI.tif
            ├── R2_Probe T5.tif
            ├── R2_Probe T6.tif
            ├── ...

    """

    abs_input_dir = os.path.abspath(input_dir)
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    for file in os.listdir(abs_input_dir):
        if file.endswith(".txt"):
            copy2(os.path.join(abs_input_dir, file), abs_output_dir)
        elif file.endswith(".tif"):
            rgb_array = np.asarray(Image.open(os.path.join(abs_input_dir, file)))
            img = Image.fromarray(np.amax(rgb_array,
                                          axis=2))  # take the max rgb value rather than ITU-R
            # 601-2 luma transform from PIL
            img.save(os.path.join(abs_output_dir, file), format='tiff')

# Uncomment to run
#convert_rgb_to_greyscale('original_data', 'RNAScope_grayscale')

###################################################################################################
# Formatting into SpaceTx Format
# ------------------------------
# Now that all the images are greyscale, we can use the TileFetcher interface and
# write_experiment_json() function to load the images and save in SpaceTx Format. We also define a
# build_codebook() function that turns the gene list.txt into a codebook.json, which is needed by
# experiment.json.

import functools
import os
from typing import Mapping, Tuple, Union

import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, Features


@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return imread(file_path)


def ch_to_Tnum(ch_label: int, round_label: int):
    """
    The file naming scheme for channel doesn't start over at '1' for each round.
    Instead it continues counting from the previous round.
    Convert channel int used by SpaceTx format to T-number used in filenames.
    """

    return 4 * (round_label) + (ch_label + 1)


class RNATile(FetchedTile):

    def __init__(
            self,
            file_path: str
    ) -> None:
        """Parser for an RNAScope tile.

        Parameters
        ----------
        file_path : str
            location of the tiff
        coordinates :
            the coordinates for the selected RNAScope tile, extracted from the metadata
        """
        self.file_path = file_path

        # dummy coordinates must match shape
        self._coordinates = {
            Coordinates.X: (0.0, 0.1120),
            Coordinates.Y: (0.0, 0.0539),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 539, Axes.X: 1120}  # hard coded for these datasets.

    @property
    def coordinates(self):
        return self._coordinates  # noqa

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)  # slice out the correct z-plane


class RNADAPITile(FetchedTile):

    def __init__(
            self,
            file_path: str
    ) -> None:
        """Parser for an RNAScope tile.

        Parameters
        ----------
        file_path : str
            location of the tiff
        coordinates :
            the coordinates for the selected RNAScope tile, extracted from the metadata
        """
        self.file_path = file_path

        # dummy coordinates must match shape
        self._coordinates = {
            Coordinates.X: (0.0, 0.1120),
            Coordinates.Y: (0.0, 0.0539),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 539, Axes.X: 1120}  # hard coded for these datasets.

    @property
    def coordinates(self):
        return self._coordinates  # noqa

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)  # slice out the correct z-plane


class RNATileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for RNAScope data.

        This TileFetcher constructs spaceTx format for a stitched tissue slice, where
        `input_dir` is a directory containing .tif files with the following structure:

        └── RNAScope_greyscale
            ├── R1_DAPI.tif
            ├── R1_Probe T1.tif
            ├── R1_Probe T2.tif
            ├── R1_Probe T3.tif
            ├── R1_Probe T4.tif
            ├── R2_DAPI.tif
            ├── R2_Probe T5.tif
            ├── R2_Probe T6.tif
            ├── ...

        Notes
        -----
        - The spatial organization of the fields of view are not known so they are filled by
        dummy coordinates
        """

        self.input_dir = input_dir
        self.num_z = 1

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = f"R{(round_label + 1)}_Probe T{ch_to_Tnum(ch_label, round_label)}.tif"
        return RNATile(os.path.join(self.input_dir, filename))


class RNADAPITileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for dapi auxiliary images of RNAScope experiment.

        Every round has a DAPI image, which can be used for image registration and cell segmentation

        """
        self.input_dir = input_dir

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = f"R{(round_label + 1)}_DAPI.tif"
        return RNADAPITile(os.path.join(self.input_dir, filename))


def build_codebook(gene_file: str, codebook_json: str) -> Codebook:
    """Writes a codebook json file that a experiment.json file can point to.

    Data provided by ACDBio included a file with genes listed in order from T1-T12 starting from
    line 3.
    Codebook will map 3 rounds and 4 channels to genes.

    Parameters
    ----------
    gene_list : str
        Parse text file with genes listed in order of Tnum
    codebook_json : str
        Save a codebook to json using SpaceTx Format.

    Returns
    -------
    Codebook :
        Codebook object in SpaceTx format.
    """
    mappings = list()
    rnd = 0
    ch = 0
    with open(gene_file, 'r') as gene_list:
        for line_num, line in enumerate(gene_list):
            if line_num > 1:
                mappings.append({Features.CODEWORD: [
                    {Axes.ROUND.value: rnd, Axes.CH.value: ch, Features.CODE_VALUE: 1}],
                                 Features.TARGET: line.strip()})
                ch = ch + 1
                if ch > 3:
                    rnd = rnd + 1
                    ch = 0

    Codebook.from_code_array(mappings).to_json(codebook_json)
    return Codebook.from_code_array(mappings)


def format_experiment(input_dir, output_dir) -> None:
    """CLI entrypoint for spaceTx format construction for RNAScope data

    Parameters
    ----------
    input_dir : str
        directory containing input data. See TileFetcher classes for expected directory structures.
    output_dir : str
        directory that 2-d images and starfish metadata will be written to.
    """
    abs_output_dir = os.path.abspath(output_dir)
    abs_input_dir = os.path.abspath(input_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    primary_tile_fetcher = RNATileFetcher(abs_input_dir)
    dapi_tile_fetcher = RNADAPITileFetcher(abs_input_dir)

    # This is hardcoded for this example data set
    primary_image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: 3,
        Axes.CH: 4,
        Axes.ZPLANE: 1,
    }

    aux_images_dimensions: Mapping[str, Mapping[Union[str, Axes], int]] = {
        "nuclei": {
            Axes.ROUND: 3,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
    }

    write_experiment_json(
        path=output_dir,
        fov_count=1,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_images_dimensions,
        primary_tile_fetcher=primary_tile_fetcher,
        aux_tile_fetcher={"nuclei": dapi_tile_fetcher},
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )

    build_codebook(os.path.join(abs_input_dir, 'list.txt'),
                   os.path.join(abs_output_dir, 'codebook.json'))


# Uncomment to run
#format_experiment('/Users/mcai/RNAScope HiPlex/RNAScope_grayscale', 'starfish_format')

###################################################################################################
# starfish pipeline
# -----------------
# To process the images with starfish first load the experiment: codebook, primary images, and DAPI
# images.

from starfish import Experiment, FieldOfView

# from starfish import Experiment, FieldOfView
experiment = Experiment.from_json('https://d26bikfyahveg8.cloudfront.net/RNAScope/HiPlex_formatted/experiment.json')
codebook = experiment.codebook
imgs = experiment["fov_000"].get_image(FieldOfView.PRIMARY_IMAGES) # primary images contain fluorescence RNA spots
dapi = experiment["fov_000"].get_image('nuclei') # dapi images conatain nuclei

###################################################################################################
# Visually check images look correct.

# Uncomment code below to visualize with napari
#%gui qt
#from starfish import display

#viewer = display(imgs)

###################################################################################################
# Image Registration
# --------------------
# Images need to be registered. Here we use dapi to learn the translation transform of images
# relative to the first round, and then apply it to imgs.

#%matplotlib inline

import matplotlib
from starfish.util.plot import diagnose_registration
from starfish.types import Axes

# Visualize alignment of dapi images across rounds
matplotlib.rcParams["figure.dpi"] = 250
diagnose_registration(dapi, {Axes.ROUND:0}, {Axes.ROUND:1}, {Axes.ROUND:2})

###################################################################################################
# We use FFT cross-correlation to find the translational shift with precision of 1/1000th of a
# pixel. No scaling or rotation is computed.

from starfish.image import LearnTransform

learn_translation = LearnTransform.Translation(reference_stack=dapi.sel({Axes.ROUND: 0}), axes=Axes.ROUND, upsampling=1000)
transforms_list = learn_translation.run(dapi)

###################################################################################################
# Now we apply the learned transform to dapi images and confirm registration was successful.

from starfish.image import ApplyTransform

warp = ApplyTransform.Warp()
registered_dapi = warp.run(dapi, transforms_list=transforms_list, in_place=False)
diagnose_registration(registered_dapi, {Axes.ROUND:0}, {Axes.ROUND:1}, {Axes.ROUND:2})

###################################################################################################
# Due to a bug, we need to repeat learn_translation.run before applying transform to primary images.

transforms_list = learn_translation.run(dapi)
registered_imgs = warp.run(imgs, transforms_list=transforms_list, in_place=False)

###################################################################################################
# Signal Enhancement and Background Reduction
# -------------------------------------------
# Representative raw primary images from each channel show RNA spots have Airy pattern in channels 1
# and 2 with faint concentric rings around discs. Channels 3 and 4 have more background, lower SNR.

import matplotlib.pyplot as plt
from starfish.util.plot import imshow_plane

f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(14, 4.5))
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax1, title='Ch 1')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 1, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax2, title='Ch 2')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 2, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax3, title='Ch 3')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 3, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax4, title='Ch 4')
f.suptitle('raw primary images')
f.tight_layout()

###################################################################################################
# The background can be removed with a white-tophat filter. The kernel has a radius larger than
# an RNA spot, so only foreground larger than RNA is removed.

from starfish.image import Filter

# white tophat filter
whitetophat = Filter.WhiteTophat(masking_radius=3, is_volume=False)
wth_imgs = whitetophat.run(registered_imgs, in_place=False)

f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(14, 4.5))
imshow_plane(wth_imgs, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax1, title='Ch 1')
imshow_plane(wth_imgs, sel={Axes.ROUND: 0, Axes.CH: 1, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax2, title='Ch 2')
imshow_plane(wth_imgs, sel={Axes.ROUND: 0, Axes.CH: 2, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax3, title='Ch 3')
imshow_plane(wth_imgs, sel={Axes.ROUND: 0, Axes.CH: 3, Axes.X: (200,400), Axes.Y: (250,450)}, ax=ax4, title='Ch 4')
f.suptitle('white tophat filtered')
f.tight_layout()

###################################################################################################
# Spot Finding with BlobDetector
# ------------------------------
# Since each gene is detected in an independent (round, channel), run BlobDetector without a
# reference_image. This will find spots in every (round, channel) image.
#
# Run with is_volume=True because of issue #1870

from starfish.spots import FindSpots

bd = FindSpots.BlobDetector(
    min_sigma=0.5,
    max_sigma=2,
    num_sigma=9,
    threshold=0.1,
    is_volume=True,
    overlap=0.1,
    measurement_type='mean',
)
bd_spots = bd.run(image_stack=wth_imgs)

###################################################################################################
# To check the SpotFindingResults with filtered fluorescence image, we first need to separate the
# results by (round, channel) and convert to numpy array. Here we store the arrays in a list.

# save SpotAttributes for each (round,channel) as numpy array in list
bd_spots_numpy = list()
for rnd in bd_spots.round_labels:
    for ch in bd_spots.ch_labels:
        bd_spots_numpy.append(bd_spots[{Axes.CH:ch, Axes.ROUND:rnd}].spot_attrs.data[['z', 'y', 'x']].to_numpy())

###################################################################################################
# We can interactively visualize the SpotFindingResults for each (round, channel) as a separate
# points layer in napari on top of the ImageStack. This is the ideal way to explore results and
# find errors.

# display found spots for each (round,channel) as a layer in napari
#viewer = display(stack=wth_imgs)
#layer_index = 0
#for rnd in bd_spots.round_labels:
#    for ch in bd_spots.ch_labels:
#        viewer.add_points(data=bd_spots_numpy[layer_index], symbol='ring', face_color='red',
#                size=5,
#                          name=f'r: {rnd}, ch: {ch}', visible=False)
#        layer_index = layer_index + 1

###################################################################################################
# For readability of this notebook in printed format, here are the SpotFindingResults plotted in
# a figure.

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows = 2, ncols=4, figsize=(14, 9))
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 0, Axes.X: (200,400),
                                   Axes.Y: (250, 450)}, ax=ax1, title='Ch 1')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 1, Axes.X: (200,400),
                                   Axes.Y: (250, 450)}, ax=ax2, title='Ch 2')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 2, Axes.X: (200,400),
                                   Axes.Y: (250, 450)}, ax=ax3, title='Ch 3')
imshow_plane(registered_imgs, sel={Axes.ROUND: 0, Axes.CH: 3, Axes.X: (200,400),
                                   Axes.Y: (250, 450)}, ax=ax4, title='Ch 4')
layer_index = 0
for ax in [ax5, ax6, ax7, ax8]:
    ax.scatter(bd_spots_numpy[layer_index][:,2], bd_spots_numpy[layer_index][:,1], s=3,
               facecolors='none', edgecolors='r')
    ax.set_xlim([200, 400])
    ax.set_ylim([250, 450])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    layer_index = layer_index + 1
f.suptitle('blobdetector spots found')
f.tight_layout()

###################################################################################################
# Decoding Spots
# --------------
# For RNAScope, decoding is simply labeling each spot with the gene that corresponds to the (round,
# channel) it was found in. Therefore, we can use SimpleLookupDecoder here.

from starfish.spots import DecodeSpots

decoder = DecodeSpots.SimpleLookupDecoder(codebook=experiment.codebook)
decoded_intensities = decoder.run(spots=bd_spots)

###################################################################################################
# Watershed Segmentation of Cells
# -------------------------------
# To then partition RNA spots into single cells, we first need to segment the image into regions
# that are defined with binary masks. Here we will use intensity thresholds, connected component
# analysis, and watershed to do the image segmentation.
#
# We take advantage of the cell background in the unfiltered primary images and treat it as a
# "cell stain" image. starfish's Segment.Watershed uses the dapi image to seed a watershed of the
# "cell stain" image, resulting in binary masks that define the area regions of the cells.

from starfish.image import Segment

# set parameters
dapi_thresh = .15  # global threshold value for nuclei images
stain_thresh = .16  # global threshold value for primary images
min_dist = 17  # minimum distance (pixels) between nuclei distance transformed peaks

seg = Segment.Watershed(
    nuclei_threshold=dapi_thresh,
    input_threshold=stain_thresh,
    min_distance=min_dist
)

# masks is BinaryMaskCollection for downstream steps
masks = seg.run(registered_imgs, registered_dapi)

# display intermediate images and result
seg.show()

###################################################################################################
# Assign Spots to Cells
# ---------------------
# Once the image is segmented into single cells, assigning RNA spots is simple.
# AssignTargets.Label labels each spot with the cell mask it is located in (or 'nan' if not
# located in a cell). Then the DecodedIntensityTable can be transformed to an ExpressionMatrix
# and viewed as a DataFrame or saved for downstream analysis.

from starfish.spots import AssignTargets

# Assign spots to cells by labeling each spot with cell_id
al = AssignTargets.Label()
labeled = al.run(masks, decoded_intensities)

# Filter out spots that are not located in any cell mask
labeled_filtered = labeled[labeled.cell_id != 'nan']

# Transform to expression matrix and show first 3 genes
mat = labeled_filtered.to_expression_matrix()
mat.to_pandas()

###################################################################################################
# Save ExpressionMatrix as .h5ad-format AnnData
# ---------------------------------------------

#mat.save_anndata('RNAScope_HiPlex_Example.h5ad')