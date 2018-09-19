from typing import Optional

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class RegistrationAlgorithmBase(AlgorithmBase):
    def run(self, stack) -> Optional[ImageStack]:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
