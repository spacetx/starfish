from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithmBase(AlgorithmBase):
    def run(self, stack: ImageStack) -> ImageStack:
        """Performs filtering on an ImageStack."""
        raise NotImplementedError()
