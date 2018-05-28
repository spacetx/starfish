from typing import Union, Tuple, Dict, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from starfish.munge import dataframe_to_multiindex


class IntensityTable(xr.DataArray):

    def __init__(self, intensities, coords: Mapping=None, dims: Tuple[str]=None,
                 name: Union[str, Tuple[str]]=None, attrs: Dict=None, encoding: Dict=None) -> None:
        """Table to store feature (spot, pixel) intensities and associated metadata across image tiles

        Parameters
        ----------
        intensities : np.ndarray[Any]
            intensity data
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates. If sequence-like, should be a sequence
            of tuples where the first element is the dimension name and the
            second element is the corresponding coordinate array_like object.
        dims : Union[str, Tuple[str]]
            Name(s) of the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        name : str
            Name of this array.
        attrs : OrderedDict[Any]
            Dictionary holding arbitrary metadata for object
        encoding : Dict
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset', 'dtype',
            'units' and 'calendar' (the later two only for datetime arrays).
            Unrecognized keys are ignored.

        """

        # TODO ambrosejcarr: make some checks here on the data
        super().__init__(data=intensities, coords=coords, dims=dims, name=name, attrs=attrs, encoding=encoding)

    @classmethod
    def from_spot_data(cls, intensity_data: pd.DataFrame, tile_data: pd.DataFrame, feature_attributes: pd.DataFrame):
        coords = (
            dataframe_to_multiindex(tile_data),
            dataframe_to_multiindex(feature_attributes)
        )

        return cls(
            intensities=intensity_data.values,
            coords=coords,
            dims=('tiles', 'features')
        )

    def show(self, background_image: np.ndarray) -> None:
        """show spots on a background image"""
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError
