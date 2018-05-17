import json

import pandas as pd

from starfish.munge import gather
from starfish.pipeline.features.encoded_spots import EncodedSpots


class SpotAttributes:

    def __init__(self, spot_attributes):
        """Construct a SpotAttributes instance

        Parameters
        ----------
        spot_attributes : pd.DataFrame

        """
        missing_fields = {'x', 'y', 'r', 'intensity', 'spot_id'}.difference(spot_attributes.columns)
        if missing_fields:
            raise ValueError('spot attributes missing {!r} required fields'.format(missing_fields))

        self.data = spot_attributes

    # TODO ambrosejcarr: why geojson in addition to json?
    def save_geojson(self, output_file_name):

        geojson = [
            {
                'properties': {'id': int(row.spot_id), 'radius': int(row.r)},
                'geometry': {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
            } for row in self.data.iterrows()  # TODO is this different from no iterrows() call?
        ]

        with open(output_file_name, 'w') as f:
            f.write(json.dumps(geojson))

    def save_json(self, output_file_name):
        self.data.to_json(output_file_name, orient='records')

    @classmethod
    def load_json(cls, file):
        return cls(pd.read_json(file, orient='records'))

    def display(self):
        pass

    def encode(self, stack):

        # create stack squeeze map
        stack.squeeze(bit_map_flag=False)  # this is side-effecting :( :(
        mapping = stack.squeeze_map
        inds = range(len(stack.squeeze_map))
        d = dict(zip(inds, self.data['intensity']))
        d['spot_id'] = range(self.data.shape[0])

        res = pd.DataFrame(d)
        res = gather(res, 'ind', 'val', inds)
        res = pd.merge(res, mapping, on='ind', how='left')

        return EncodedSpots(res)
