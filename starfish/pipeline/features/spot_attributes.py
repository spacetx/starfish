import json

import numpy as np
import pandas as pd

from ._validated_table import ValidatedTable


class SpotAttributes(ValidatedTable):

    required_fields = {
        'x',  # spot x-coordinate
        'y',  # spot y-coordinate
        'r',  # spot radius
        'intensity',  # intensity of spot (commonly max or average)
        'spot_id'  # integer spot id
    }

    def __init__(self, spot_attributes: pd.DataFrame) -> None:
        """Construct a SpotAttributes instance

        Parameters
        ----------
        spot_attributes : pd.DataFrame

        """
        super().__init__(spot_attributes, SpotAttributes.required_fields)

    def save_geojson(self, output_file_name: str) -> None:
        """Save to geojson for web visualization

        Parameters
        ----------
        output_file_name : str
            name of output json file

        """

        # TODO ambrosejcarr: write a test for this
        geojson = [
            {
                'properties': {'id': int(row.spot_id), 'radius': int(row.r)},
                'geometry': {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
            } for index, row in self.data.iterrows()
        ]

        with open(output_file_name, 'w') as f:
            f.write(json.dumps(geojson))

    def display(self, background_image: np.ndarray) -> None:
        """

        Parameters
        ----------
        background_image : np.ndarray
            image on which to plot spots. If 3d, take a max projection

        Returns
        -------

        """
        # TODO ambrosejcarr: re-implement this from the show() method
        pass
