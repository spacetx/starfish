from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.types import LOG
from starfish.types._constants import STARFISH_EXTRAS_KEY
from starfish.util.logging import LogEncoder


class AlgorithmBaseType(type):

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(bases) != 0:
            # this is _not_ AlgorithmBase.  Instead, it's a subclass of AlgorithmBase.
            cls.run = AlgorithmBaseType.run_with_logging(cls.run)

    @staticmethod
    def run_with_logging(func):
        """
        This method extends each pipeline component.run() method to also log itself and
        runtime parameters to the IntensityTable and ImageStack objects. There are two
        scenarios for this method:
            1.) Filtering:
                    ImageStack -> ImageStack
            2.) Spot Detection:
                    ImageStack -> IntensityTable
                    ImageStack -> [IntensityTable, ConnectedComponentDecodingResult]
            TODO segmentation and decoding
        """
        def helper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Scenario 1, Filtering
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

    PipelineComponent: `starfish.image._segmentation.Segmentation(PipelineComponent)`

    AlgorithmBase: `starfish.image._segmentation._base.SegmentationAlgorithmBase(AlgorithmBase)`

    Implementing Algorithms:
    - `starfish.image._segmentation.watershed.Watershed(SegmentationAlgorithmBase)`

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
