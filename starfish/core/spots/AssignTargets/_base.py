from abc import abstractmethod

import numpy as np

from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class AssignTargetsAlgorithm(metaclass=AlgorithmBase):
    """
    AssignTargets assigns cell IDs to detected spots using an IntensityTable and
    SegmentationMaskCollection.
    """

    @abstractmethod
    def run(
            self,
            label_image: np.ndarray,
            decoded_intensity_table: DecodedIntensityTable,
            verbose: bool=False,
            in_place: bool=False,
    ) -> DecodedIntensityTable:
        """Performs target (e.g. gene) assignment given the spots and the regions."""
        raise NotImplementedError()
