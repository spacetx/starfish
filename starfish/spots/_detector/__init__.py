import argparse
import os
from typing import Type

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from starfish.types import Indices
from starfish.util.argparse import FsExistsType
from . import _base
from . import gaussian
from . import pixel_spot_detector
from . import trackpy_local_max_peak_finder


class SpotFinder(PipelineComponent):

    spot_finder_group: argparse.ArgumentParser

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.SpotFinderAlgorithmBase

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the spot finder component to the CLI argument parser."""
        spot_finder_group = subparsers.add_parser("detect_spots")
        spot_finder_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        spot_finder_group.add_argument("-o", "--output", required=True)
        spot_finder_group.add_argument(
            '--blobs-stack', default=None, required=False, help=(
                'ImageStack that contains the blobs. Will be max-projected across imaging round '
                'and channel to produce the blobs_image'
            )
        )
        spot_finder_group.add_argument(
            '--reference-image-from-max-projection', default=False, action='store_true', help=(
                'Construct a reference image by max projecting imaging rounds and channels. Spots '
                'are found in this image and then measured across all images in the input stack.'
            )
        )
        spot_finder_group.add_argument(
            '--codebook', default=None, required=False, help=(
                'A spaceTx spec-compliant json file that describes a three dimensional tensor '
                'whose values are the expected intensity of a spot for each code in each imaging '
                'round and each color channel.'
            )
        )
        spot_finder_group.set_defaults(starfish_command=SpotFinder._cli)
        spot_finder_subparsers = spot_finder_group.add_subparsers(
            dest="spot_finder_algorithm_class")

        for algorithm_cls in cls._algorithm_to_class_map().values():
            group_parser = spot_finder_subparsers.add_parser(algorithm_cls._get_algorithm_name())
            group_parser.set_defaults(spot_finder_algorithm_class=algorithm_cls)
            algorithm_cls._add_arguments(group_parser)

        cls.spot_finder_group = spot_finder_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the spot finder component based on parsed arguments."""

        if args.spot_finder_algorithm_class is None or print_help:
            cls.spot_finder_group.print_help()
            cls.spot_finder_group.exit(status=2)

        print('Detecting Spots ...')
        image_stack = ImageStack.from_path_or_url(args.input)

        if args.codebook is not None:
            args.codebook = Codebook.from_json(args.codebook)

        instance = args.spot_finder_algorithm_class(**vars(args))

        if args.blobs_stack is not None:
            blobs_stack = ImageStack.from_path_or_url(args.blobs_stack)  # type: ignore
            blobs_image = blobs_stack.max_proj(Indices.ROUND, Indices.CH)
            intensities = instance.run(
                image_stack,
                blobs_image=blobs_image,
                reference_image_from_max_projection=args.reference_image_from_max_projection
            )
        else:
            intensities = instance.run(image_stack)

        # When PixelSpotDetector is used run() returns a tuple
        if isinstance(intensities, tuple):
            intensities = intensities[0]
        intensities.save(args.output)
