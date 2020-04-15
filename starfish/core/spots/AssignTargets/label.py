import warnings

import numpy as np

from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.types import Axes, Features
from ._base import AssignTargetsAlgorithm


class Label(AssignTargetsAlgorithm):
    """
    Extract cell ids for features in IntensityTable from a set of segmentation masks.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @classmethod
    def _add_arguments(cls, parser) -> None:
        pass

    @staticmethod
    def _assign(
        masks: BinaryMaskCollection,
        decoded_intensities: DecodedIntensityTable,
        in_place: bool,
    ) -> DecodedIntensityTable:

        cell_ids = (
            Features.AXIS,
            np.full(decoded_intensities.sizes[Features.AXIS], fill_value='nan', dtype='<U8')
        )

        decoded_intensities[Features.CELL_ID] = cell_ids

        # it's 3D data.
        for _, mask in masks:
            has_z_data = Axes.ZPLANE.value in mask.coords
            if has_z_data:
                z_min, z_max = float(mask.z.min()), float(mask.z.max())
            else:
                warnings.warn(
                    "AssignTargets will require 3D masks in the future.", DeprecationWarning)
                z_min, z_max = np.NINF, np.inf
            y_min, y_max = float(mask.y.min()), float(mask.y.max())
            x_min, x_max = float(mask.x.min()), float(mask.x.max())

            in_bbox = decoded_intensities.where(
                (decoded_intensities.z >= z_min)
                & (decoded_intensities.z <= z_max)
                & (decoded_intensities.y >= y_min)
                & (decoded_intensities.y <= y_max)
                & (decoded_intensities.x >= x_min)
                & (decoded_intensities.x <= x_max),
                drop=True
            )

            selectors = {'y': in_bbox.y, 'x': in_bbox.x}
            if has_z_data:
                selectors['z'] = in_bbox.z
            in_mask = mask.sel(**selectors)
            spot_ids = in_bbox[Features.SPOT_ID][in_mask.values]
            decoded_intensities[Features.CELL_ID].loc[
                decoded_intensities[Features.SPOT_ID].isin(spot_ids)] = mask.name

        return decoded_intensities

    def run(
            self,
            masks: BinaryMaskCollection,
            decoded_intensity_table: DecodedIntensityTable,
            verbose: bool = False,
            in_place: bool = False,
    ) -> DecodedIntensityTable:
        """Extract cell ids for features in IntensityTable from a segmentation label image

        Parameters
        ----------
        masks : BinaryMaskCollection
            binary masks segmenting each cell
        decoded_intensity_table : IntensityTable
            spot information
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on the percentage completed during processing (default = False)

        Returns
        -------
        IntensityTable :
            IntensityTable with added features variable containing cell ids. Points outside of
            cells will be assigned `nan`.

        """
        return self._assign(masks, decoded_intensity_table, in_place=in_place)
