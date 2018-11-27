

class AlgorithmBaseType(type):

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(bases) != 0:
            # this is _not_ PipelineComponent.  Instead, it's a subclass of PipelineComponent.
            cls.run = AlgorithmBaseType.run_with_logging(cls.run)

    @staticmethod
    def run_with_logging(func):
        def helper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, tuple):
                result.update_log(args[0])
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



