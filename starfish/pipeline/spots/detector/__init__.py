import os
import json

from starfish.pipeline.pipeline_component import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import _base
from starfish.munge import spots_to_geojson

from . import gaussian


class SpotFinder(PipelineComponent):

    spot_finder_group = None

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

        # todo lazy imports go here

        if args.spot_finder_algorithm_class is None or print_help:
            cls.spot_finder_group.print_help()
            cls.spot_finder_group.exit(status=2)

        instance = args.spot_finder_algorithm_class.from_cli_args(args)

        spots_df_tidy = instance.detect(**vars(args))

        if args.show:
            instance.show(figsize=(10, 10))

        spots_viz = instance.spots_df_viz
        geojson = spots_to_geojson(spots_viz)

        path = os.path.join(args.output, 'spots.geojson')
        print("Writing | spots geojson to: {}".format(path))
        with open(path, 'w') as f:
            f.write(json.dumps(geojson))

        path = os.path.join(args.output, 'spots.json')
        print("Writing | spot_id | x | y | z | to: {}".format(path))
        spots_viz.to_json(path, orient="records")

        path = os.path.join(args.output, 'encoder_table.json')
        print("Writing | spot_id | hyb | ch | val | to: {}".format(path))
        spots_df_tidy.to_json(path, orient="records")


SpotFinder._ensure_algorithms_setup()