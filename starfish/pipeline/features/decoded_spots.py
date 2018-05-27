import pandas as pd
from ._validated_table import ValidatedTable


class DecodedSpots(ValidatedTable):

    required_fields = {
        # TODO ambrosejcarr change barcode -> codeword, gene -> gene_name (requires rewrite of codebook)
        'barcode',  # the code word that corresponds to this gene
        'quality',  # spot quality
        'spot_id',  # integer spot id
        'gene'  # string gene name
    }

    def __init__(self, decoded_spots: pd.DataFrame) -> None:
        """

        Parameters
        ----------
        decoded_spots : pd.DataFrame

        """
        # call the validation routine, set self.data = required_fields
        super().__init__(decoded_spots, DecodedSpots.required_fields)
