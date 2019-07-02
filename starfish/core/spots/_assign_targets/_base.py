from abc import abstractmethod
from typing import Type

import numpy as np

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent


class AssignTargets(PipelineComponent):
    pass


class AssignTargetsAlgorithm(AlgorithmBase):
    """
    AssignTargets assigns cell IDs to detected spots using an IntensityTable and
    SegmentationMaskCollection.
    """
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return AssignTargets

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
