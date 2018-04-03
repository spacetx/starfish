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
        import pandas

        # TODO this should be loaded from disk
        d = {'barcode': ['AAGC', 'AGGC'], 'gene': ['ACTB_human', 'ACTB_mouse']}
        codebook = pandas.DataFrame(d)
        decoder = iss.IssDecoder(codebook, letters=['T', 'G', 'C', 'A'])

        return decoder.decode(encoded)
