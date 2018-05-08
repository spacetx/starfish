from starfish.decoders import iss
from ._base import DecoderAlgorithmBase


class IssDecoder(DecoderAlgorithmBase):
    @classmethod
    def from_cli_args(cls, args):
        return IssDecoder()

    @classmethod
    def get_algorithm_name(cls):
        return "iss"

    @classmethod
    def add_arguments(cls, group_parser):
        pass

    def decode(self, encoded, codebook):
        decoder = iss.IssDecoder(codebook, letters=['T', 'G', 'C', 'A'])

        return decoder.decode(encoded)
