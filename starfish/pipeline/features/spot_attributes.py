import json

import pandas as pd


from starfish.pipeline.features.encoded_spots import EncodedSpots


class SpotAttributes:

    def __init__(self, spot_attributes):
        """Construct a SpotAttributes instance

        Parameters
        ----------
        spot_attributes : pd.DataFrame

        """
        required_fields = {
            'x',  # spot x-coordinate
            'y',  # spot y-coordinate
            'r',  # spot radius
            'intensity',  # intensity of spot (commonly max or average)
            'spot_id'  # integer spot id
        }
        missing_fields = required_fields.difference(spot_attributes.columns)
        if missing_fields:
            raise ValueError(f'spot attributes missing {missing_fields:!r} required fields')

        self.data = spot_attributes

    # TODO ambrosejcarr: why geojson in addition to json?
    def save_geojson(self, output_file_name):

        # TODO ambrosejcarr: write a test for this
        geojson = [
            {
                'properties': {'id': int(row.spot_id), 'radius': int(row.r)},
                'geometry': {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
            } for index, row in self.data.iterrows()
        ]

        with open(output_file_name, 'w') as f:
            f.write(json.dumps(geojson))

    def save(self, output_file_name):
        self.data.to_json(output_file_name, orient='records')

    @classmethod
    def load(cls, file):
        return cls(pd.read_json(file))

    def display(self, background_image):
        """

        Parameters
        ----------
        background_image : np.ndarray
            image on which to plot spots. If 3d, take a max projection

        Returns
        -------

        """
        # TODO re-implement this from the show() method
        pass
