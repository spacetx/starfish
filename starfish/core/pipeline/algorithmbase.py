import functools
import inspect
from abc import ABCMeta

from starfish.core.types._constants import STARFISH_EXTRAS_KEY


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
        """
        @functools.wraps(func)
        def helper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result is not None:
                method_class_str = str(args[0].__class__)
                if 'ApplyTransform' in method_class_str or 'Filter' in method_class_str:
                    # Update the log on the resulting ImageStack
                    result.log.update_log(args[0])
                if 'FindSpots' in method_class_str:
                    # Update the log on the resulting SpotFindingResults
                    result.log.update_log(args[0])
                if 'DecodeSpots' in method_class_str:
                    # update log then transfer to DecodedIntensityTable
                    spot_results = kwargs['spots']
                    spot_results.log.update_log(args[0])
                    result.attrs[STARFISH_EXTRAS_KEY] = spot_results.log.encode()
                if 'DetectPixels' in method_class_str:
                    stack = args[1]
                    # update log with spot detection instance args[0]
                    stack.log.update_log(args[0])
                    # get resulting intensity table and set log
                    it = result[0]
                    it.attrs[STARFISH_EXTRAS_KEY] = stack.log.encode()
            return result
        return helper
