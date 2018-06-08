import argparse
import os

from starfish.io import Stack
from starfish.pipeline.features.pixels import _base
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import pixel_spot_detector


class PixelSpotDetector(PipelineComponent):

    pixel_finder_algorithm_class = None
    pixel_finder_group: argparse.ArgumentParser

    @classmethod
    def implementing_algorithms(cls):
        return _base.PixelFinderAlgorithmBase.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the pixel finder component to the CLI argument parser."""
        pixel_finder_group = subparsers.add_parser("detect_pixels")
        pixel_finder_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        pixel_finder_group.add_argument("-o", "--output", required=True)
        pixel_finder_group.set_defaults(starfish_command=PixelSpotDetector._cli)
        pixel_finder_subparsers = pixel_finder_group.add_subparsers(dest="pixel_finder_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = pixel_finder_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(pixel_finder_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.pixel_finder_group = pixel_finder_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the pixel finder component based on parsed arguments."""

        if args.pixel_finder_algorithm_class is None or print_help:
            cls.pixel_finder_group.print_help()
            cls.pixel_finder_group.exit(status=2)

        print('Detecting Pixels...')
        s = Stack()
        s.read(args.input)
        instance = args.pixel_finder_algorithm_class(**vars(args))
        # TODO does this work?
        pixel_attributes, encoded_pixels = instance.find(s)

        # TODO ambrosejcarr: this still needs to be added back.
        # if args.show:
        #     encoded_pixels.show(figsize=(10, 10))

        path = os.path.join(args.output, 'pixels.geojson')
        print(f"Writing | pixels geojson to: {path}")
        pixel_attributes.save_geojson(path)

        path = os.path.join(args.output, 'pixels.json')
        print(f"Writing | spot_id | x | y | z | to: {path}")
        pixel_attributes.save(path)

        path = os.path.join(args.output, 'encoder_table.json')
        print(f"Writing | spot_id | hyb | ch | val | to: {path}")
        encoded_pixels.save(path)
