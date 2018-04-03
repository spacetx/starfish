class DecoderAlgorithmBase(object):
    @classmethod
    def from_cli_args(cls, args):
        """
        Given parsed arguments, construct an instance of this decoder algorithm.

        Generally, this involves retrieving the appropriate attributes from args to pass to the decoder algorithm's
        constructor.
        """
        raise NotImplementedError()

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

    def decode(self, encoded, codebook):
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
