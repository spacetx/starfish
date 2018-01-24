import collections

from ..util.argparse import FsExistsType
from . import _base
from . import _fourier_shift


class Registration(object):
    algorithm_to_class_map = dict()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the registration component to the CLI argument parser."""
        register_group = subparsers.add_parser("register")
        register_group.add_argument("in_json", type=FsExistsType())
        register_group.add_argument("out_dir", type=FsExistsType())
        register_group.set_defaults(starfish_command=Registration._cli)
        registration_subparsers = register_group.add_subparsers(dest="registration_algorithm_class")

        cls._ensure_algorithms_setup()
        for algorithm_cls in cls.algorithm_to_class_map.values():
            algorithm_cls.add_to_parser(registration_subparsers)

        cls.register_group = register_group

    @classmethod
    def run(cls, algorithm_name, stack, *args, **kwargs):
        """Runs the registration component using the algorithm name, stack, and arguments for the specific algorithm."""
        cls._ensure_algorithms_setup()
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
        s.read(args.in_json)

        instance.register(s)

        s.write(args.out_dir)

    @classmethod
    def _ensure_algorithms_setup(cls):
        if len(cls.algorithm_to_class_map) != 0:
            return

        queue = collections.deque(_base.RegistrationAlgorithmBase.__subclasses__())
        while len(queue) > 0:
            algorithm_cls = queue.popleft()
            queue.extend(algorithm_cls.__subclasses__())

            cls.algorithm_to_class_map[algorithm_cls.__name__] = algorithm_cls
