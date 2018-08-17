from typing import Optional

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.stack import ImageStack


class FilterAlgorithmBase(AlgorithmBase):
    def run(self, stack: ImageStack) -> Optional[ImageStack]:
        """Performs filtering on an ImageStack."""
        raise NotImplementedError()
