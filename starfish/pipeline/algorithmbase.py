from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable


class AlgorithmBaseType(type):

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(bases) != 0:
            # this is _not_ AlgorithmBase.  Instead, it's a subclass of AlgorithmBase.
            cls.run = AlgorithmBaseType.run_with_logging(cls.run)

    @staticmethod
    def run_with_logging(func):
        def helper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Spot detection returns a tuple or IntensityTable, filtering returns an Imagestack
            if isinstance(result, ImageStack):
                result.update_log(args[0])
            # Spot detection
            elif isinstance(result, tuple) or isinstance(result, IntensityTable):
                if isinstance(args[1], ImageStack):
                    stack = args[1]
                    # update log with spot detection instance args[0]
                    stack.update_log(args[0])
                    # get resulting intensity table and set log
                    it = result
                    if isinstance(result, tuple):
                        it = result[0]
                    it.attrs.update({'log': stack.log})
            return result
        return helper


class AlgorithmBase(metaclass=AlgorithmBaseType):
    @classmethod
    def _get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        return cls.__name__
