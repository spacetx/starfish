import argparse
import os

from starfish.image import ImageStack
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import _base
from . import gaussian
from . import local_max_peak_finder


class SpotFinder(PipelineComponent):

    spot_finder_group: argparse.ArgumentParser

    @classmethod
    def implementing_algorithms(cls):
        return _base.SpotFinderAlgorithmBase.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the spot finder component to the CLI argument parser."""
        spot_finder_group = subparsers.add_parser("detect_spots")
        spot_finder_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        spot_finder_group.add_argument("-o", "--output", required=True)
        spot_finder_group.set_defaults(starfish_command=SpotFinder._cli)
        spot_finder_subparsers = spot_finder_group.add_subparsers(dest="spot_finder_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = spot_finder_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(spot_finder_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.spot_finder_group = spot_finder_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the spot finder component based on parsed arguments."""

        if args.spot_finder_algorithm_class is None or print_help:
            cls.spot_finder_group.print_help()
            cls.spot_finder_group.exit(status=2)

        print('Detecting Spots ...')
        hybridization_stack = ImageStack.from_path_or_url(args.input)
        instance = args.spot_finder_algorithm_class(**vars(args))
        spot_attributes, encoded_spots = instance.find(hybridization_stack)

        if args.show:
            encoded_spots.show(figsize=(10, 10))

        path = os.path.join(args.output, 'spots.geojson')
        print(f"Writing | spots geojson to: {path}")
        spot_attributes.save_geojson(path)

        path = os.path.join(args.output, 'spots.json')
        print(f"Writing | spot_id | x | y | z | to: {path}")
        spot_attributes.save(path)

        path = os.path.join(args.output, 'encoder_table.json')
        print(f"Writing | spot_id | hyb | ch | val | to: {path}")
        encoded_spots.save(path)
