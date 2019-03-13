from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class ApplyTransformBase(AlgorithmBase):
    def run(self, stack) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
