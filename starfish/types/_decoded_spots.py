import json
from typing import Iterable

import numpy as np
import pandas as pd

from starfish.types import Axes, Features
from ._validated_table import ValidatedTable


class DecodedSpots(ValidatedTable):

    required_fields = {
        Axes.X.value,          # spot x-coordinate
        Axes.Y.value,          # spot y-coordinate
        Features.TARGET,     # spot z-coordinate
    }

    def __init__(self, decoded_spots: pd.DataFrame) -> None:
        """Construct a decoded_spots instance

        Parameters
        ----------
        decoded_spots : pd.DataFrame

        """
        super().__init__(decoded_spots, DecodedSpots.required_fields)

    def save_geojson(self, output_file_name: str) -> None:
        """Save to geojson for web visualization

        Parameters
        ----------
        output_file_name : str
            name of output json file

        """

        geojson = [
            {
                'properties': {'id': int(row.spot_id), 'radius': int(row.r)},
                'geometry': {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
            } for index, row in self.data.iterrows()
        ]

        with open(output_file_name, 'w') as f:
            f.write(json.dumps(geojson))

    def save_csv(self, output_file_name: str) -> None:
        self.data.to_csv(output_file_name, index=False)

    @classmethod
    def load_csv(cls, file_name: str) -> "DecodedSpots":
        return cls(pd.read_csv(file_name))
