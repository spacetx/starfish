"""
.. _tutorial_assigning_spots:

Assigning Spots to Cells
========================

In a starfish pipeline, creating a single cell gene expression matrix is a two step process. It
requires a :py:class:`.DecodedIntensityTable` with :term:`features <Feature>` mapped to
:term:`targets <Target>` and a :py:class:`.BinaryMaskCollection` with cell masks that were found
by :ref:`segmenting cells in the FOV <section_segmenting_cells>`.

The first step is to run :py:class:`.Label` to label each feature in the
:py:class:`.DecodedIntensityTable` with the ``cell_id`` of the cell mask the feature is located in.
Features that are not within the boundaries of any cell mask are labeled ``nan`` in the ``cell_id``
column. Features that are in multiple cell masks (i.e. the cell masks overlap) will be assigned to
the last mask in the :py:class:`.BinaryMaskCollection` that the feature is found in.

The second step is to transform the :py:class:`.DecodedIntensityTable` into an
:py:class:`.ExpressionMatrix` with :py:meth:`.to_expression_matrix`. At this stage, additional
cell metadata can be added. The :py:class:`.ExpressionMatrix` can then be saved in various
popular formats for single-cell RNAseq analysis packages.

This tutorial demonstrates how to use decoded MERFISH spots and the
:ref:`manually segmented <tutorial_manual_segmentation>` cell masks to assign spots to cells and
create a gene expression matrix. It then provides a couple examples how cell metadata can be
added to the :py:class:`.ExpressionMatrix` and how the matrix can be saved for loading into other
analysis tools.
"""

# Load MERFISH data
import os
import numpy as np
from copy import deepcopy
from starfish import BinaryMaskCollection, data
from starfish.core.experiment.experiment import FieldOfView
from starfish.image import Filter
from starfish.spots import DetectPixels, AssignTargets
from starfish.types import Axes, Features, Levels

experiment = data.MERFISH()
fov = experiment["fov_000"]
imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
dapi = fov.get_image("nuclei")  # nuclei

# filter and deconvolve data
ghp = Filter.GaussianHighPass(sigma=3)
dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
glp = Filter.GaussianLowPass(sigma=1)
ghp.run(imgs, in_place=True)
dpsf.run(imgs, in_place=True)
glp.run(imgs, in_place=True)

# scale data with user-defined factors to normalize images. For this data set, the scale factors
# are stored in experiment.json.
scale_factors = {
    (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
    for t in experiment.extras['scale_factors']
}
filtered_imgs = deepcopy(imgs)
for selector in imgs._iter_axes():
    data = filtered_imgs.get_slice(selector)[0]
    scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
    filtered_imgs.set_slice(selector, scaled, [Axes.ZPLANE])

# Decode with PixelSpotDecoder
psd = DetectPixels.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean',
    norm_order=2,
    distance_threshold=0.5176,
    magnitude_threshold=1.77e-5,
    min_area=2,
    max_area=np.inf,
)
initial_spot_intensities, prop_results = psd.run(filtered_imgs)
# Select only decoded spots that pass thresholds and map to genes in codebook
decoded = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]
decoded_filtered = decoded[decoded.target != 'nan']

# Load cell mask
roi_path = os.path.join(os.path.dirname("__file__"), 'RoiSet.zip')
# Uncomment the following code to download RoiSet.zip
#import requests, shutil
#response = requests.get("https://raw.github.com/spacetx/starfish/mcai-segmentation-docs/examples/RoiSet.zip", stream=True)
#response.raw.decode_content = True
#with open(roi_path, "wb") as f:
#    shutil.copyfileobj(response.raw, f)
masks = BinaryMaskCollection.from_fiji_roi_set(path_to_roi_set_zip=roi_path, original_image=dapi)

# Assign spots to cells by labeling each spot with cell_id
al = AssignTargets.Label()
labeled = al.run(masks, decoded_filtered)

# Filter out spots that are not located in any cell mask
labeled_filtered = labeled[labeled.cell_id != 'nan']

###################################################################################################
# Now that every :term:`feature <Feature>` in the :py:class:`.DecodedIntensityTable` is labeled
# with a valid ``cell_id``, the features can be grouped by cell into a single cell gene expression
# matrix. In this matrix, each row is a cell and each column is a gene. The values within the
# matrix are the number of features of that particular gene in that particular cell.

# Transform to expression matrix and show first 12 genes
mat = labeled_filtered.to_expression_matrix()
mat.to_pandas().iloc[:, 0:12].astype(int)

###################################################################################################
# In addition to the matrix, :py:class:`.ExpressionMatrix` contains cell metadata, (e.g. cell
# location and cell size) stored as ``Coordinates`` of the matrix. When transforming a
# :py:class:`.DecodedIntensityTable` to an :py:class:`.ExpressionMatrix`, the initial cell metadata
# ``Coordinates`` are the location, number of undecoded spots, and area. The location is not
# based on the cell masks, but calculated from the central position of spots assigned to each
# cell. The number of undecoded spots is zero for each cell in this example because the undecoded
# spots were removed after decoding. The area, on the other hand, is always set to zero and needs
# to be calculated from the :py:class:`.BinaryMaskCollection` as shown below. New metadata fields
# can also be added.

# Add area (in pixels) of cell masks to expression matrix metadata
mat[Features.AREA] = (Features.CELLS, [mask.data.sum() for _, mask in masks])

# Add eccentricity of cell masks to expression matrix metadata
from skimage.measure import regionprops
mat['ecc'] = (Features.CELLS, [regionprops(mask.data.astype(int), coordinates='rc')[0].eccentricity for _, mask in masks])

# Show expression matrix with metadata
mat

# Hierarchically cluster matrix and view as heatmap
import seaborn as sns
sns.clustermap(mat.data.T,
            yticklabels=mat.genes.data,
            xticklabels=['cell {}'.format(n + 1) for n in range(25)],
            cmap='magma')

###################################################################################################
# Finally, the :py:class:`.ExpressionMatrix` can be loaded into other analysis tools such as
# `scanpy`_ and `Seurat`_ by saving it as an AnnData, Loom, or NetCDF file.
#
# .. _scanpy: https://scanpy.readthedocs.io/en/stable/api/index.html#reading
# .. _Seurat: https://satijalab.org/loomR/loomR_tutorial.html
#
# Save as .netcdf for saving and loading in starfish pipeline
mat.save('expression_matrix.nc')

# Save as .h5ad file for loading in scanpy
mat.save_anndata('expression_matrix.h5ad')

# Save as .loom file for loading with loompy or loomR
mat.save_loom('expression_matrix.loom')
