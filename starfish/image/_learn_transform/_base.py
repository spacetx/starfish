from starfish.image._learn_transform.transforms_list import TransformsList
from starfish.pipeline.algorithmbase import AlgorithmBase


class LearnTransformBase(AlgorithmBase):
    def run(self, stack, axis) -> TransformsList:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
