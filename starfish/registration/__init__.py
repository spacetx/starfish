import collections
import functools

from ..util.argparse import FsExistsType
from . import _base
from . import _fourier_shift


class Registration(object):
    algorithm_to_class_map = dict()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the registration component to the CLI argument parser."""
        register_group = subparsers.add_parser("register")
        register_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        register_group.add_argument("-o", "--output", type=FsExistsType(), required=True)
        register_group.set_defaults(starfish_command=Registration._cli)
        registration_subparsers = register_group.add_subparsers(dest="registration_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map.values():
            group_parser = registration_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(registration_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.register_group = register_group

    @classmethod
    def run(cls, algorithm_name, stack, *args, **kwargs):
        """Runs the registration component using the algorithm name, stack, and arguments for the specific algorithm."""
        algorithm_cls = cls.algorithm_to_class_map[algorithm_name]
        instance = algorithm_cls(*args, **kwargs)
        return instance.register(stack)

    @classmethod
    def _cli(cls, args):
        """Runs the registration component based on parsed arguments."""
        if args.registration_algorithm_class is None:
            cls.register_group.print_help()
            cls.register_group.exit(status=2)

        instance = args.registration_algorithm_class.from_cli_args(args)

        from ..io import Stack

        print('Registering ...')
        s = Stack()
        s.read(args.input)

        instance.register(s)

        s.write(args.output)

    @classmethod
    def _ensure_algorithms_setup(cls):
        if len(cls.algorithm_to_class_map) != 0:
            return

        queue = collections.deque(_base.RegistrationAlgorithmBase.__subclasses__())
        while len(queue) > 0:
            algorithm_cls = queue.popleft()
            queue.extend(algorithm_cls.__subclasses__())

            cls.algorithm_to_class_map[algorithm_cls.__name__] = algorithm_cls


Registration._ensure_algorithms_setup()
