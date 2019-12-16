from abc import abstractmethod
from typing import Sequence

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class MergeAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(
            self,
            binary_mask_collections: Sequence[BinaryMaskCollection],
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        """Merge multiple binary mask collections together."""
        raise NotImplementedError()
