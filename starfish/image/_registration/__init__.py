import argparse
from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from starfish.util.argparse import FsExistsType
from . import fourier_shift
from ._base import RegistrationAlgorithmBase


class Registration(PipelineComponent):

    register_group: argparse.ArgumentParser

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return RegistrationAlgorithmBase

    @classmethod
    def _add_to_parser(cls, subparsers):
        """Adds the registration component to the CLI argument parser."""
        register_group = subparsers.add_parser("registration")
        register_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        register_group.add_argument("-o", "--output", required=True)
        register_group.set_defaults(starfish_command=Registration._cli)
        registration_subparsers = register_group.add_subparsers(dest="registration_algorithm_class")

        for algorithm_cls in cls._algorithm_to_class_map().values():
            group_parser = registration_subparsers.add_parser(algorithm_cls._get_algorithm_name())
            group_parser.set_defaults(registration_algorithm_class=algorithm_cls)
            algorithm_cls._add_arguments(group_parser)

        cls.register_group = register_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the registration component based on parsed arguments."""
        if args.registration_algorithm_class is None or print_help:
            cls.register_group.print_help()
            cls.register_group.exit(status=2)

        print('Registering ...')
        stack = ImageStack.from_path_or_url(args.input)
        instance = args.registration_algorithm_class(**vars(args))
        instance.run(stack)

        stack.write(args.output)
