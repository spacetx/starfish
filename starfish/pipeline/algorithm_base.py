class AlgorithmBase(object):

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_cli_args(cls, args):
        """
        Given parsed arguments, construct an instance of this spot finder algorithm.

        Generally, this involves retrieving the appropriate attributes from args to pass to the spot finder algorithm's
        constructor.
        """
        return cls(**vars(args))

    @classmethod
    def get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, group_parser):
        """Adds the arguments for the algorithm."""
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        """Detects spots."""
        raise NotImplementedError()
