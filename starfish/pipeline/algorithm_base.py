from starfish.image import ImageBase


class AlgorithmBase(object):
    """This class supports a set of image analysis algorithms

    """

    @classmethod
    def from_cli_args(cls, args):
        """Given parsed command line arguments, construct an instance of this algorithm."""
        raise NotImplementedError

    @classmethod
    def get_algorithm_name(cls):
        """Returns the name of the algorithm.

        This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, group_parser):
        """Adds the arguments for the algorithm."""
        raise NotImplementedError

    def run(self, stack: ImageBase):
        """run the algorithm on an Image or ImageStack

        Parameters
        ----------
        stack : starfish.image.ImageBase

        Returns
        -------

        """
        raise NotImplementedError
