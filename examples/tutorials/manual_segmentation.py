"""
.. _tutorial_manual_segmentation:

Loading ImageJ ROI Sets
=======================

.. note::
    Starfish only supports importing 2D ROIs.

In order to create a cell by gene expression matrix from image-based transcriptomics data, RNA
spots must be assigned to cells by segmenting an image. The best quality cell segmentation
annotations are manually drawn by experts. If you have ROI sets exported with `ROI manager
<https://imagej.net/docs/guide/146-30.html#fig:The-ROI-Manager>`_ in ImageJ or FIJI, they can be
loaded into starfish as a :py:class:`.BinaryMaskCollection`. The ROI set for each field of view
must be passed with the corresponding :py:class:`.ImageStack` to :py:meth:`.from_fiji_roi_set` in
order to assign accurate ``pixel ticks`` and ``physical ticks`` to the
:py:class:`.BinaryMaskCollection`.

This tutorial demonstrates how to load an ROI set that was exported form ROI manager as a .zip
and how to plot the resulting :py:class:`.BinaryMaskCollection`.

Brief segmentation workflow in ImageJ ROI Manager:

* Tools > ROI Manager
* Click "polygon selection" (third button from left on GUI)
* Create a polygon, then click the "+" button to finalize it
* Repeat until segmented
* Click "more >>>"
* Click "save"
* Save the ROISet
"""

import matplotlib
import matplotlib.pyplot as plt
import os

import starfish.data
from starfish import BinaryMaskCollection
from starfish.image import Filter
from starfish.types import Levels

matplotlib.rcParams["figure.dpi"] = 150

# Load MERFISH data to get dapi ImageStack
experiment = starfish.data.MERFISH()
fov = experiment["fov_000"]
dapi = fov.get_image("nuclei")  # nuclei


# Preprocess MERFISH dapi imagestack
def preprocess(dapi):
    blur = Filter.GaussianLowPass(sigma=5)
    blurred = blur.run(dapi)

    clip = Filter.Clip(p_min=1, p_max=95, level_method=Levels.SCALE_BY_CHUNK)
    clipped = clip.run(blurred)
    return clipped


dapi = preprocess(dapi)

# Import RoiSet.zip as BinaryMaskCollection
roi_path = os.path.join(os.path.dirname("__file__"), 'RoiSet.zip')
# Uncomment the following code to download RoiSet.zip
#import requests, shutil
#response = requests.get("https://raw.github.com/spacetx/starfish/mcai-segmentation-docs/examples/RoiSet.zip", stream=True)
#response.raw.decode_content = True
#with open(roi_path, "wb") as f:
#    shutil.copyfileobj(response.raw, f)
masks = BinaryMaskCollection.from_fiji_roi_set(path_to_roi_set_zip=roi_path, original_image=dapi)

# display BinaryMaskCollection next to original dapi image
f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(dapi.xarray.values.squeeze())
ax1.set_title("Dapi")
ax2.imshow(masks.to_label_image().xarray.values.squeeze(), cmap=plt.cm.nipy_spectral)
ax2.set_title("Imported ROI")
f.tight_layout()
