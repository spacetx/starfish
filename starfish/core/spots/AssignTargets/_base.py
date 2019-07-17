from abc import abstractmethod

import numpy as np

from starfish.core.intensity_table.intensity_table import IntensityTable
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
            intensity_table: IntensityTable,
            verbose: bool=False,
            in_place: bool=False,
    ) -> IntensityTable:
        """Performs target (e.g. gene) assignment given the spots and the regions."""
        raise NotImplementedError()
