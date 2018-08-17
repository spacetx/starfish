from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.stack import ImageStack


class RegistrationAlgorithmBase(AlgorithmBase):
    def run(self, stack) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
