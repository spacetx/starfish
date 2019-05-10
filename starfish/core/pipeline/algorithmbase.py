import functools
import inspect
from abc import ABCMeta, abstractmethod
from typing import Type

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import LOG
from starfish.core.types._constants import STARFISH_EXTRAS_KEY
from starfish.core.util.logging import LogEncoder
from .pipelinecomponent import PipelineComponent


class AlgorithmBaseType(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not inspect.isabstract(cls):
            AlgorithmBaseType.register_with_pipeline_component(cls)
            cls.run = AlgorithmBaseType.run_with_logging(cls.run)

    @staticmethod
    def register_with_pipeline_component(algorithm_cls):
        pipeline_component_cls = algorithm_cls.get_pipeline_component_class()
        if pipeline_component_cls._algorithm_to_class_map_int is None:
            pipeline_component_cls._algorithm_to_class_map_int = {}
        pipeline_component_cls._algorithm_to_class_map_int[algorithm_cls.__name__] = algorithm_cls
        setattr(pipeline_component_cls, algorithm_cls._get_algorithm_name(), algorithm_cls)

        pipeline_component_cls._cli.add_command(algorithm_cls._cli)

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

    See Also
    --------
    starfish.pipeline.pipelinecomponent.py
    """
    @classmethod
    def _get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        return cls.__name__

    @classmethod
    @abstractmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        """
        Returns the class of PipelineComponent this algorithm implements.
        """
        raise NotImplementedError()
