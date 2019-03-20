import collections
import importlib
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Set, Type

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
            PipelineComponentType._cli_register(cls)
            PipelineComponentType._register_pipeline_component_type_name(cls)

    _pipeline_component_type_name_to_class_map: MutableMapping[str, Type["PipelineComponent"]] = \
        dict()

    @classmethod
    def _register_pipeline_component_type_name(mcs, cls: Type["PipelineComponent"]) -> None:
        PipelineComponentType._pipeline_component_type_name_to_class_map[
            cls.pipeline_component_type_name()] = cls

    @staticmethod
    def get_pipeline_component_type_by_name(name: str) -> Type["PipelineComponent"]:
        return PipelineComponentType._pipeline_component_type_name_to_class_map[name]

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

    @classmethod
    def _cli_register(mcs, cls):
        for algorithm_cls in cls._algorithm_to_class_map().values():
            cls._cli.add_command(algorithm_cls._cli)


class PipelineComponent(metaclass=PipelineComponentType):
    """
    This is the base class for any method executed by the CLI.

    PipelineComponent is an Abstract Class that exposes two private methods to link any subclassing
    method to the CLI, _algorithm_to_class_map, which fetches all the algorithms that extend this
    base class at run time, and _cli_register, which registers those methods to the CLI. It exposes
    two additional abstract private methods that must be extended by subclasses:

    Methods
    -------
    _get_algorithm_base_class()
        should simply return an instance of the AlgorithmBase. See, e.g.
        starfish.image.segmentation.Segmentation
    _cli_run(ctx, instance, *args, **kwargs)
        implements the behavior of the pipeline component that must occur when this component is
        evoked from the CLI. This often includes loading serialized objects into memory and
        passing them to the API's run command.

    See Also
    --------
    starfish.pipeline.algorithmbase.py
    """

    _algorithm_to_class_map_int: Optional[Mapping[str, Type]] = None

    @staticmethod
    def get_pipeline_component_class_by_name(name: str) -> Type["PipelineComponent"]:
        return PipelineComponentType.get_pipeline_component_type_by_name(name)

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        """
        Returns the name of the pipeline component type.
        """
        raise NotImplementedError()

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        """
        Get the base class that algorithms which implement this pipeline stage must extend.
        Pipeline components must provide this method.
        """
        raise NotImplementedError()

    @classmethod
    def _algorithm_to_class_map(cls) -> Mapping[str, Type]:
        """Returns a mapping from algorithm names to the classes that implement them."""
        assert cls._algorithm_to_class_map_int is not None
        return cls._algorithm_to_class_map_int

    @classmethod
    def _cli_run(cls, ctx, instance, *args, **kwargs):
        raise NotImplementedError()


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
        excluded = {"__init__.py"}

    path: Path = Path(path_str).parent
    for entry in path.iterdir():
        if not entry.suffix.lower().endswith(".py"):
            continue
        if entry.name.lower() in excluded:
            continue

        importlib.import_module(f".{entry.stem}", package)
