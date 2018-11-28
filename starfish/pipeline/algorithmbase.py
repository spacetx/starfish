from starfish.imagestack.imagestack import ImageStack


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
            # Spot detection returns a tuple, filtering returns an Imagestack
            if isinstance(result, ImageStack):
                # Only run on second run of function if in_place is True
                in_place = 'in_place' in kwargs and kwargs['in_place'] is True
                if not in_place:
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
