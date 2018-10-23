import argparse
from typing import Type

from skimage.io import imsave

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from starfish.util.argparse import FsExistsType
from . import watershed
from ._base import SegmentationAlgorithmBase


class Segmentation(PipelineComponent):

    segmentation_group: argparse.ArgumentParser

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return SegmentationAlgorithmBase

    @classmethod
    def _add_to_parser(cls, subparsers) -> None:
        """Adds the segmentation component to the CLI argument parser."""
        segmentation_group = subparsers.add_parser("segment")
        segmentation_group.add_argument("--hybridization-stack", type=FsExistsType(), required=True)
        segmentation_group.add_argument("--nuclei-stack", type=FsExistsType(), required=True)
        segmentation_group.add_argument("-o", "--output", required=True)
        segmentation_group.set_defaults(starfish_command=Segmentation._cli)
        segmentation_subparsers = segmentation_group.add_subparsers(
            dest="segmentation_algorithm_class")

        for algorithm_cls in cls._algorithm_to_class_map().values():
            group_parser = segmentation_subparsers.add_parser(algorithm_cls._get_algorithm_name())
            group_parser.set_defaults(segmentation_algorithm_class=algorithm_cls)
            algorithm_cls._add_arguments(group_parser)

        cls.segmentation_group = segmentation_group

    @classmethod
    def _cli(cls, args, print_help=False) -> None:
        """Runs the segmentation component based on parsed arguments."""
        if args.segmentation_algorithm_class is None or print_help:
            cls.segmentation_group.print_help()
            cls.segmentation_group.exit(status=2)

        instance = args.segmentation_algorithm_class(**vars(args))

        print('Segmenting ...')
        hybridization_stack = ImageStack.from_path_or_url(args.hybridization_stack)
        nuclei_stack = ImageStack.from_path_or_url(args.nuclei_stack)
        label_image = instance.run(hybridization_stack, nuclei_stack)

        print(f"Writing label image to {args.output}")
        imsave(args.output, label_image)
