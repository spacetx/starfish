import json
from typing import Collection, Sequence

import numpy as np
import pandas as pd

from starfish.core.types import Axes, Features
from ._validated_table import ValidatedTable


class SpotAttributes(ValidatedTable):

    required_fields = [
        (Axes.X.value, int),            # spot x-coordinate
        (Axes.Y.value, int),            # spot y-coordinate
        (Axes.ZPLANE.value, int),       # spot z-coordinate
        (Features.SPOT_RADIUS, float)   # spot radius
    ]

    def __init__(self, spot_attributes: pd.DataFrame) -> None:
        """Construct a SpotAttributes instance

        Parameters
        ----------
        spot_attributes : pd.DataFrame

        """
        super().__init__(spot_attributes, {i[0] for i in SpotAttributes.required_fields})

    @classmethod
    def empty(cls, extra_fields: Collection = tuple()) -> "SpotAttributes":
        """return an empty SpotAttributes object"""
        extra_dtypes: list = list(zip(extra_fields, [object] * len(extra_fields)))
        dtype = cls.required_fields + extra_dtypes
        return cls(pd.DataFrame(np.array([], dtype=dtype)))

    @classmethod
    def combine(cls, spot_attribute_tables: Sequence["SpotAttributes"]) -> "SpotAttributes":
        return cls(pd.concat([
            spot_attribute_table.data
            for spot_attribute_table in spot_attribute_tables
        ]))

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
