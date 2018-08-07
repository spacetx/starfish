from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class RegistrationAlgorithmBase(AlgorithmBase):
    def run(self, stack) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
