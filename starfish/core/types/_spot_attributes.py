import json
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from starfish.core.types import Axes, Coordinates, Features
from ._validated_table import ValidatedTable


class SpotAttributes(ValidatedTable):

    required_fields = {
        Axes.X.value,          # spot x-coordinate
        Axes.Y.value,          # spot y-coordinate
        Axes.ZPLANE.value,     # spot z-coordinate
        Features.SPOT_RADIUS,  # spot radius
        # Features.INTENSITY,    # spot intensity
        # Features.SPOT_ID,      # spot ID
        # Coordinates.X.value,   # physical x
        # Coordinates.Y.value,   # physical y
        # Coordinates.Z.value    # physical Z
    }

    def __init__(self, spot_attributes: pd.DataFrame) -> None:
        """Construct a SpotAttributes instance

        Parameters
        ----------
        spot_attributes : pd.DataFrame

        """
        super().__init__(spot_attributes, SpotAttributes.required_fields)

    def assign_physical_coordinates(self, imagestack):
        pairs = (
            (Axes.X.value, Coordinates.X.value),
            (Axes.Y.value, Coordinates.Y.value),
            (Axes.ZPLANE.value, Coordinates.Z.value)
        )
        for axis, coord in pairs:
            imagestack_pixels: xr.DataArray = imagestack.xarray[axis]
            spot_pixel_offsets: np.ndarray = self.data[axis].values

            # can't interpolate if the axis size == 1, so just select in that case.
            if len(imagestack_pixels) == 1:
                coordinate_fetcher = imagestack_pixels.sel
            else:
                coordinate_fetcher = imagestack_pixels.interp
            coordinates = coordinate_fetcher({axis: spot_pixel_offsets})[coord]

            self.data[coord] = coordinates.values


    @classmethod
    def empty(cls, extra_fields: Iterable = tuple()) -> "SpotAttributes":
        """return an empty SpotAttributes object"""
        fields = list(cls.required_fields.union(extra_fields))
        dtype = list(zip(fields, [np.object] * len(fields)))
        return cls(pd.DataFrame(np.array([], dtype=dtype)))

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
