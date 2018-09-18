import sys

from validate_sptx.validate_sptx import validate


class Cli:
    parser_group = None

    @staticmethod
    def add_to_parser(parser):
        """adds experiment-json and fuzz arguments to the given parser"""
        parser.add_argument(
            "--experiment-json",
            required=True,
            metavar="JSON_FILE_OR_URL")
        parser.add_argument(
            "--fuzz", action="store_true")
        parser.set_defaults(starfish_command=Cli.run)
        Cli.parser_group = parser

    @staticmethod
    def run(args, print_help=False):
        """invokes validate with the parsed commandline arguments"""
        try:
            valid = validate(args.experiment_json, fuzz=args.fuzz)
            if valid:
                sys.exit(0)
            else:
                sys.exit(1)
        except KeyboardInterrupt:
            sys.exit(3)
