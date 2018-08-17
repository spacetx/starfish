import argparse
import json
from typing import MutableMapping

from starfish.types import Indices
from starfish.util.argparse import FsExistsType
from . import AUX_IMAGE_NAMES, write_experiment_json


class StarfishIndex:
    def __call__(self, spec_json):
        try:
            spec = json.loads(spec_json)
        except json.decoder.JSONDecodeError:
            raise argparse.ArgumentTypeError(
                "Could not parse {} into a valid index specification.".format(spec_json))

        return {
            Indices.ROUND: spec.get(Indices.ROUND, 1),
            Indices.CH: spec.get(Indices.CH, 1),
            Indices.Z: spec.get(Indices.Z, 1),
        }


class Cli:
    parser_group = None

    """
    Maps the name of an aux image to the name in the parsed arguments object.
    """
    name_arg_map: MutableMapping[str, str] = dict()

    @staticmethod
    def add_to_parser(parser):
        parser.add_argument(
            "output_dir",
            type=FsExistsType())
        parser.add_argument(
            "--fov-count",
            type=int,
            required=True,
            help="Number of FOVs in this experiment.")
        parser.add_argument(
            "--hybridization-dimensions",
            type=StarfishIndex(),
            required=True,
            help="Dimensions for the hybridization images.  Should be a json dict, with {}, {}, "
                 "and {} as the possible keys.  The value should be the shape along that "
                 "dimension.  If a key is not present, the value is assumed to be 0.".format(
                Indices.ROUND.value,
                Indices.CH.value,
                Indices.Z.value))
        for aux_image_name in AUX_IMAGE_NAMES:
            arg = parser.add_argument(
                "--{}-dimensions".format(aux_image_name),
                type=StarfishIndex(),
                help="Dimensions for the {} images.  Should be a json dict, with {}, {}, and {} as "
                     "the possible keys.  The value should be the shape along that dimension.  If "
                     "a key is not present, the value is assumed to be 0.".format(
                    aux_image_name, Indices.ROUND, Indices.CH, Indices.Z))
            Cli.name_arg_map[aux_image_name] = arg.dest
        parser.set_defaults(starfish_command=Cli.run)

        Cli.parser_group = parser

    @staticmethod
    def run(args, print_help=False):
        write_experiment_json(
            args.output_dir, args.fov_count, args.hybridization_dimensions,
            {
                aux_image_name: getattr(args, Cli.name_arg_map[aux_image_name])
                for aux_image_name in AUX_IMAGE_NAMES
            }
        )
