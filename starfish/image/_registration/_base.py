from typing import Optional

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.stack import ImageStack


class RegistrationAlgorithmBase(AlgorithmBase):
    def run(self, stack) -> Optional[ImageStack]:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
