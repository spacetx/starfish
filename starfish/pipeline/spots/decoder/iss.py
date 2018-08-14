from ._base import DecoderAlgorithmBase


class IssDecoder(DecoderAlgorithmBase):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def add_arguments(cls, group_parser):
        pass

    def decode(self, intensities, codebook):
        return codebook.decode_per_round_max(intensities)
