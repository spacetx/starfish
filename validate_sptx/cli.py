import argparse
import json

from starfish.util.argparse import FsExistsType
from validate_sptx.validate_sptx import validate


class Cli:
    parser_group = None

    @staticmethod
    def add_to_parser(parser):
        parser.add_argument(
            "--experiment-json",
            required=True,
            metavar="JSON_FILE_OR_URL",
            type=FsExistsType())
        parser.add_argument(
            "--fuzz", action="store_true")
        parser.set_defaults(starfish_command=Cli.run)
        Cli.parser_group = parser

    @staticmethod
    def run(args, print_help=False):
        validate(args.experiment_json, fuzz=args.fuzz)
