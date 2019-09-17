from abc import abstractmethod

from starfish.core.image._registration.transforms_list import TransformsList
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class LearnTransformAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, stack, *args) -> TransformsList:
        """Learns Transforms for a given stack."""
        raise NotImplementedError()
