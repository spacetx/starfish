from pathlib import Path
from typing import Union

import h5py
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology.Binarize import ThresholdBinarize
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from ._base import SegmentAlgorithm


class IllastikPretrained(SegmentAlgorithm):
    """
    Import pretrained illastik model as ImageStack then run ThresholdBinarize
    """

    def __init__(self, path_to_ilastic_model: Union[str, Path], threshold=None):
        h5 = h5py.File(path_to_ilastic_model)
        self.probability_images = h5["exported_data"][:]
        h5.close()
        self.theshold = threshold

    def run(self,
            primary_images: ImageStack,
            nuclei: ImageStack,
            *args
            ) -> BinaryMaskCollection:

        cell_probabilities, background_probabilities = self.probability_images[:, :,
                                                       0], self.probability_images[:, :, 1]
        cell_threshold = threshold_otsu(cell_probabilities)
        label_array = ndi.label(cell_probabilities > cell_threshold)[0]
        image_stack = ImageStack.from_numpy(label_array)
        binarizer = ThresholdBinarize(threshold=self.theshold)
        return binarizer.run(image_stack)
