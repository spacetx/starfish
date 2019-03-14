from starfish.image._learn_transform.transforms_list import TransformsList
from starfish.pipeline.algorithmbase import AlgorithmBase


class LearnTransformBase(AlgorithmBase):
    def run(self, stack) -> TransformsList:
        """Learns Transforms for a given stack."""
        raise NotImplementedError()
