import functools
import importlib
import inspect
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Set

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import LOG
from starfish.core.types._constants import STARFISH_EXTRAS_KEY
from starfish.core.util.logging import LogEncoder


class AlgorithmBaseType(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not inspect.isabstract(cls):
            cls.run = AlgorithmBaseType.run_with_logging(cls.run)

    @staticmethod
    def run_with_logging(func):
        """
        This method extends each pipeline component.run() method to also log itself and
        runtime parameters to the IntensityTable and ImageStack objects. There are two
        scenarios for this method:
            1.) Filtering/ApplyTransform:
                    Imagestack -> Imagestack
            2.) Spot Detection:
                    ImageStack -> IntensityTable
                    ImageStack -> [IntensityTable, ConnectedComponentDecodingResult]
            TODO segmentation and decoding
        """
        @functools.wraps(func)
        def helper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Scenario 1, Filtering, ApplyTransform
            if isinstance(result, ImageStack):
                result.update_log(args[0])
            # Scenario 2, Spot detection
            elif isinstance(result, tuple) or isinstance(result, IntensityTable):
                if isinstance(args[1], ImageStack):
                    stack = args[1]
                    # update log with spot detection instance args[0]
                    stack.update_log(args[0])
                    # get resulting intensity table and set log
                    it = result
                    if isinstance(result, tuple):
                        it = result[0]
                    it.attrs[STARFISH_EXTRAS_KEY] = LogEncoder().encode({LOG: stack.log})
            return result
        return helper


class AlgorithmBase(metaclass=AlgorithmBaseType):

    """
    This is the base class of any algorithm that starfish exposes.

    Subclasses of this base class are paired with subclasses of PipelineComponent. The subclasses of
    PipelineComponent retrieve subclasses of the paired AlgorithmBase. Together, the two classes
    enable starfish to expose a paired API and CLI.

    Examples
    --------

    PipelineComponent: `starfish.image._segment.Segmentation(PipelineComponent)`

    AlgorithmBase: `starfish.image._segment._base.SegmentationAlgorithmBase(AlgorithmBase)`

    Implementing Algorithms:
    - `starfish.image._segment.watershed.Watershed(SegmentationAlgorithmBase)`

    This pattern exposes the API as follows:

    `starfish.image.Segmentation.<implementing algorithm (Watershed)>`

    and the CLI as:

    `$> starfish segmentation watershed`

    To create an entirely new group of related algorithms, like `Segmentation`, a new subclass of
    both `AlgorithmBase` and `PipelineComponent` must be created.

    To add to an existing group of algorithms like "Segmentation", an algorithm implementation must
    subclass the corresponding subclass of `AlgorithmBase`. In this case,
    `SegmentationAlgorithmBase`.
    """

    @classmethod
    def _get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        return cls.__name__

    @abstractmethod
    def run(self):
        raise NotImplementedError


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
