import argparse
import collections
from typing import Mapping, Optional, Type


class PipelineComponentType(type):
    """
    This is the metaclass for PipelineComponent.  As each subclass that is _not_ PipelineComponent is created, it sets
    up a map between the algorithm name and the class that implements it.
    """
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(bases) != 0:
            # this is _not_ PipelineComponent.  Instead, it's a subclass of PipelineComponent.
            PipelineComponentType._ensure_algorithms_setup(cls)

    @classmethod
    def _ensure_algorithms_setup(mcs, cls):
        if cls._algorithm_to_class_map is None:
            cls._algorithm_to_class_map = dict()

            queue = collections.deque(cls.implementing_algorithms())
            while len(queue) > 0:
                algorithm_cls = queue.popleft()
                queue.extend(algorithm_cls.__subclasses__())

                cls._algorithm_to_class_map[algorithm_cls.__name__] = algorithm_cls

                setattr(cls, algorithm_cls.get_algorithm_name(), algorithm_cls)


class PipelineComponent(metaclass=PipelineComponentType):

    _algorithm_to_class_map: Optional[Mapping[str, Type]] = None

    @classmethod
    def implementing_algorithms(cls):
        """
        Get a list of classes that implement an algorithm relevant to this pipeline stage.  Pipeline components must
        provide this method.
        """
        raise NotImplementedError()

    @classmethod
    def algorithm_to_class_map(cls):
        """Returns a mapping from algorithm names to the classes that implement them."""
        return cls._algorithm_to_class_map

    @classmethod
    def run(cls, algorithm_name, stack, *args, **kwargs):
        """Runs the registration component using the algorithm name, stack, and arguments for the specific algorithm."""
        algorithm_cls = cls._algorithm_to_class_map[algorithm_name]
        instance = algorithm_cls(*args, **kwargs)
        return instance.register(stack)

    @classmethod
    def _cli(cls, args: argparse.Namespace):
        raise NotImplementedError()
