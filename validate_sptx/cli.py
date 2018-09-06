import argparse
import json

from starfish.util.argparse import FsExistsType
from .validate_sptx import validate_sptx


class Cli:
    parser_group = None

    @staticmethod
    def add_to_parser(parser):
        parser.add_argument(
            "--experiment-json",
            type=FsExistsType())
        parser.add_argument(
            "--fuzz", action="store_true")
        parser.set_defaults(starfish_command=Cli.run)
        Cli.parser_group = parser

    @staticmethod
    def run(args, print_help=False):
        validate_sptx(args.experiment_json, fuzz=args.fuzz)
