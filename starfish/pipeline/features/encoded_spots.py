import pandas as pd

from starfish.constants import Indices
from ._validated_table import ValidatedTable


class EncodedSpots(ValidatedTable):

    required_fields = {
        'spot_id',  # integer spot id
        'barcode_index',  # position in barcode
        'intensity',  # spot intensity
        Indices.CH.value,  # channel
        Indices.HYB.value,  # hybridization round
    }

    def __init__(self, encoded_spots: pd.DataFrame) -> None:
        """

        Parameters
        ----------
        encoded_spots : pd.DataFrame

        """
        super().__init__(encoded_spots, EncodedSpots.required_fields)
