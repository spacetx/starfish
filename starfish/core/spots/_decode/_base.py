from abc import abstractmethod
from typing import Type

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent


class Decode(PipelineComponent):
    """
    The Decode class exposes methods to compare detected spots to expected patterns of
    fluorescence across the rounds and channels of an experiment and map them to target genes or
    proteins.

    For single molecule FISH or RNAscope experiments, these codebooks are often simple mappings of
    (round, channel) pairs to targets. For coded assays, these codebooks can be much more complex.

    Example codebooks are associated with each experiment in :py:mod:`starfish.data` and can
    be accessed with :py:meth`Experiment.codebook`.
    """
    pass


class DecodeAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Decode

    @abstractmethod
    def run(self, intensities: IntensityTable, *args):
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
