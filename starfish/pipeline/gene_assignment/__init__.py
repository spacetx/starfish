import argparse
import json

from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import _base
from . import point_in_poly


class GeneAssignment(PipelineComponent):

    gene_assignment_group: argparse.ArgumentParser

    @classmethod
    def implementing_algorithms(cls):
        return _base.GeneAssignmentAlgorithm.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the gene_assignment component to the CLI argument parser."""
        gene_assignment_group = subparsers.add_parser("gene_assignment")
        gene_assignment_group.add_argument("--coordinates-geojson", type=FsExistsType(), required=True)
        gene_assignment_group.add_argument("--spots-json", type=FsExistsType(), required=True)
        gene_assignment_group.add_argument("-o", "--output", required=True)
        gene_assignment_group.set_defaults(starfish_command=GeneAssignment._cli)
        gene_assignment_subparsers = gene_assignment_group.add_subparsers(dest="gene_assignment_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = gene_assignment_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(gene_assignment_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.gene_assignment_group = gene_assignment_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the gene_assignment component based on parsed arguments."""
        import pandas
        from starfish import munge

        if args.gene_assignment_algorithm_class is None or print_help:
            cls.gene_assignment_group.print_help()
            cls.gene_assignment_group.exit(status=2)

        with open(args.coordinates_geojson, "r") as fh:
            coordinates = json.load(fh)
        regions = munge.geojson_to_region(coordinates)

        spots = pandas.read_json(args.spots_json, orient="records")

        instance = args.gene_assignment_algorithm_class(**vars(args))

        result = instance.assign_genes(spots, regions)

        print("Writing | cell_id | spot_id to: {}".format(args.output))
        result.to_json(args.output, orient="records")
