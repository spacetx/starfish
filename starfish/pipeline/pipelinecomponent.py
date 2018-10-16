import argparse
import collections
from typing import Mapping, Optional, Type

from .algorithmbase import AlgorithmBase


class PipelineComponentType(type):
    """
    This is the metaclass for PipelineComponent.  As each subclass that is _not_ PipelineComponent
    is created, it sets up a map between the algorithm name and the class that implements it.
    """
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(bases) != 0:
            # this is _not_ PipelineComponent.  Instead, it's a subclass of PipelineComponent.
            PipelineComponentType._ensure_algorithms_setup(cls)

    @classmethod
    def _ensure_algorithms_setup(mcs, cls):
        if cls._algorithm_to_class_map_int is None:
            cls._algorithm_to_class_map_int = dict()

            queue = collections.deque(cls._get_algorithm_base_class().__subclasses__())
            while len(queue) > 0:
                algorithm_cls = queue.popleft()
                queue.extend(algorithm_cls.__subclasses__())

                cls._algorithm_to_class_map_int[algorithm_cls.__name__] = algorithm_cls

                setattr(cls, algorithm_cls._get_algorithm_name(), algorithm_cls)


class PipelineComponent(metaclass=PipelineComponentType):

    _algorithm_to_class_map_int: Optional[Mapping[str, Type]] = None

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        """
        Get the base class that algorithms which implement this pipeline stage must extend.
        Pipeline components must provide this method.
        """
        raise NotImplementedError()

    @classmethod
    def _algorithm_to_class_map(cls):
        """Returns a mapping from algorithm names to the classes that implement them."""
        return cls._algorithm_to_class_map_int

    @classmethod
    def _cli(cls, args: argparse.Namespace):
        raise NotImplementedError()
