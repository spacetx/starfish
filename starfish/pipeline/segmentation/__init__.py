import json
import argparse

from starfish.image import ImageStack
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import watershed
from ._base import SegmentationAlgorithmBase


class Segmentation(PipelineComponent):

    segmentation_group: argparse.ArgumentParser

    @classmethod
    def implementing_algorithms(cls):
        return SegmentationAlgorithmBase.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the segmentation component to the CLI argument parser."""
        segmentation_group = subparsers.add_parser("segment")
        segmentation_group.add_argument("--hybridization-stack", type=FsExistsType(), required=True)
        segmentation_group.add_argument("--nuclei-stack", type=FsExistsType(), required=True)
        segmentation_group.add_argument("-o", "--output", required=True)
        segmentation_group.set_defaults(starfish_command=Segmentation._cli)
        segmentation_subparsers = segmentation_group.add_subparsers(dest="segmentation_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = segmentation_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(segmentation_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.segmentation_group = segmentation_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the segmentation component based on parsed arguments."""
        if args.segmentation_algorithm_class is None or print_help:
            cls.segmentation_group.print_help()
            cls.segmentation_group.exit(status=2)

        instance = args.segmentation_algorithm_class(**vars(args))

        print('Segmenting ...')
        hybridization_stack = ImageStack.from_path_or_url(args.hybridization_stack)
        nuclei_stack = ImageStack.from_path_or_url(args.nuclei_stack)
        regions = instance.segment(hybridization_stack, nuclei_stack)
        geojson = regions_to_geojson(regions, use_hull=False)

        print("Writing | regions geojson to: {}".format(args.output))
        with open(args.output, "w") as f:
            f.write(json.dumps(geojson))


def regions_to_geojson(r, use_hull=True):
    """Convert region geometrical data to geojson format"""

    def make_dict(id_, verts):
        d = dict()
        c = list(map(lambda x: list(x), list(map(lambda v: [int(v[0]), int(v[1])], verts))))
        d["properties"] = {"id": id_}
        d["geometry"] = {"type": "Polygon", "coordinates": c}
        return d

    if use_hull:
        coordinates = r.hull
    else:
        coordinates = r.coordinates
    return [make_dict(id_, verts) for id_, verts in enumerate(coordinates)]
