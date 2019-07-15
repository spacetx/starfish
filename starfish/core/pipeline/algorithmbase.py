import functools
import inspect
from abc import ABCMeta

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import LOG
from starfish.core.types._constants import STARFISH_EXTRAS_KEY
from starfish.core.util.logging import LogEncoder


class AlgorithmBase(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not inspect.isabstract(cls):
            cls.run = AlgorithmBase.run_with_logging(cls.run)

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
