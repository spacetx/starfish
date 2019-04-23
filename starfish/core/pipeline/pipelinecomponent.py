import importlib
from abc import abstractmethod
from pathlib import Path
from typing import Mapping, Optional, Set, Type


class PipelineComponent:
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

    @classmethod
    def _algorithm_to_class_map(cls) -> Mapping[str, Type]:
        """Returns a mapping from algorithm names to the classes that implement them."""
        assert cls._algorithm_to_class_map_int is not None
        return cls._algorithm_to_class_map_int

    @classmethod
    @abstractmethod
    def _cli_run(cls, ctx, instance):
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
