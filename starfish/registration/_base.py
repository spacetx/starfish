class RegistrationAlgorithmBase(object):
    @classmethod
    def from_cli_args(cls, args):
        """
        Given parsed arguments, construct an instance of this registration algorithm.

        Generally, this involves retrieving the appropriate attributes from args to pass to the registration algorithm's
        constructor.
        """
        raise NotImplementedError()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the registration algorithm to the CLI argument parser."""
        raise NotImplementedError()

    def register(self, stack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
