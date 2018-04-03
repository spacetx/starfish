from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util.argparse import FsExistsType
from . import _base
from . import _iss


class Decoder(PipelineComponent):
    @classmethod
    def implementing_algorithms(cls):
        return _base.DecoderAlgorithmBase.__subclasses__()

    @classmethod
    def add_to_parser(cls, subparsers):
        """Adds the decoder component to the CLI argument parser."""
        decoder_group = subparsers.add_parser("decode")
        decoder_group.add_argument("-i", "--input", type=FsExistsType(), required=True)
        decoder_group.add_argument("-o", "--output", required=True)
        decoder_group.add_argument("-c", "--codebook", type=FsExistsType(), required=True)
        decoder_group.set_defaults(starfish_command=Decoder._cli)
        decoder_subparsers = decoder_group.add_subparsers(dest="decoder_algorithm_class")

        for algorithm_cls in cls.algorithm_to_class_map().values():
            group_parser = decoder_subparsers.add_parser(algorithm_cls.get_algorithm_name())
            group_parser.set_defaults(decoder_algorithm_class=algorithm_cls)
            algorithm_cls.add_arguments(group_parser)

        cls.decoder_group = decoder_group

    @classmethod
    def _cli(cls, args, print_help=False):
        """Runs the decoder component based on parsed arguments."""
        import pandas

        if args.decoder_algorithm_class is None or print_help:
            cls.decoder_group.print_help()
            cls.decoder_group.exit(status=2)

        instance = args.decoder_algorithm_class.from_cli_args(args)

        encoded = pandas.read_json(args.input, orient="records")
        codebook = pandas.read_json(args.codebook, orient="records")

        results = instance.decode(encoded, codebook)
        print("Writing | spot_id | gene_id to: {}".format(args.output))
        results.to_json(args.output, orient="records")


Decoder._ensure_algorithms_setup()
