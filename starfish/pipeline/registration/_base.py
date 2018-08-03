from starfish.pipeline.algorithmbase import AlgorithmBase


class RegistrationAlgorithmBase(AlgorithmBase):
    def run(self, stack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
