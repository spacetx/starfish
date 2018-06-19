import argparse

from starfish.image import ImageStack
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import _base
from . import bandpass
from . import clip
from . import gaussian_high_pass
from . import gaussian_low_pass
from . import richardson_lucy_deconvolution
from . import white_tophat


class Filter(PipelineComponent):

    filter_group: argparse.ArgumentParser

    @classmethod
    def implementing_algorithms(cls):
        return _base.FilterAlgorithmBase.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the filter component to the CLI argument parser."""
        filter_group = subparsers.add_parser("filter")
        filter_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        filter_group.add_argument("-o", "--output", required=True)
        filter_group.set_defaults(starfish_command=Filter._cli)
        filter_subparsers = filter_group.add_subparsers(dest="filter_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = filter_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(filter_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.filter_group = filter_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the filter component based on parsed arguments."""

        if args.filter_algorithm_class is None or print_help:
            cls.filter_group.print_help()
            cls.filter_group.exit(status=2)

        print('Filtering images ...')
        stack = ImageStack.from_path_or_url(args.input)
        instance = args.filter_algorithm_class(**vars(args))
        instance.filter(stack)

        stack.write(args.output)
