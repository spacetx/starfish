import pandas as pd

from starfish.core.types import Coordinates, Features
from ._validated_table import ValidatedTable


class DecodedSpots(ValidatedTable):

    required_fields = {
        Coordinates.X.value,          # spot x-coordinate
        Coordinates.Y.value,          # spot y-coordinate
        Features.TARGET,     # spot gene target
    }

    def __init__(self, decoded_spots: pd.DataFrame) -> None:
        """Construct a decoded_spots instance

        Parameters
        ----------
        decoded_spots : pd.DataFrame

        """
        super().__init__(decoded_spots, DecodedSpots.required_fields)

    def save_csv(self, output_file_name: str) -> None:
        self.data.to_csv(output_file_name, index=False)

    @classmethod
    def load_csv(cls, file_name: str) -> "DecodedSpots":
        return cls(pd.read_csv(file_name))
