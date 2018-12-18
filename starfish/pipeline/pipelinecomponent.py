import collections
import importlib
from pathlib import Path
from typing import Mapping, Optional, Set, Type

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
    def _cli_run(cls, ctx, instance, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _cli_register(cls):
        for algorithm_cls in cls._algorithm_to_class_map().values():
            cls._cli.add_command(algorithm_cls._cli)


def import_all_submodules(path_str: str, package: str, excluded: Optional[Set[str]]=None) -> None:
    """
    Given a path of a __init__.py file, find all the .py files in that directory and import them
    relatively to a package.

    Parameters
    ----------
    path_str : str
        The path of a __init__.py file.
    package : str
        The package name that the modules should be imported relative to.
    excluded : Optional[Set[str]]
        A set of files not to include.  If this is not provided, it defaults to set("__init__.py").
    """
    if excluded is None:
        excluded = set("__init__.py")

    path: Path = Path(path_str).parent
    for entry in path.iterdir():
        if not entry.suffix.lower().endswith(".py"):
            continue
        if entry.name.lower() in excluded:
            continue

        importlib.import_module(f".{entry.stem}", package)
