from starfish.pipeline.algorithmbase import AlgorithmBase


class RegistrationAlgorithmBase(AlgorithmBase):
    def register(self, stack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
