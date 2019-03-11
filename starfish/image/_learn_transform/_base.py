from typing import Any, List, Tuple
from skimage.transform._geometric import GeometricTransform

from starfish.pipeline.algorithmbase import AlgorithmBase


class LearnTransformBase(AlgorithmBase):
    def run(self, stack) ->  List[Tuple[Any, GeometricTransform]]:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
