from typing import Any, Dict

import numpy as np
import regional


def spots_to_geojson(spots_viz):
    '''
    Convert spot geometrical data to geojson format
    '''

    def make_dict(row):
        row = row[1]
        d = dict()
        d['properties'] = {'id': int(row.spot_id), 'radius': int(row.r)}
        d['geometry'] = {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
        return d

    return [make_dict(row) for row in spots_viz.iterrows()]


def geojson_to_region(geojson: Dict[Any, Any]) -> regional.many:
    """
    Convert geojson data to region geometrical data.
    """
    def make_region(geometry):
        assert geometry['geometry']['type'] == "Polygon"
        region = [(coordinates[0], coordinates[1])
                  for coordinates in geometry['geometry']['coordinates']]

        return regional.one(region)

    return regional.many([make_region(geometry) for geometry in geojson])
