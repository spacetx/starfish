from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr

from starfish.munge import dataframe_to_multiindex
from starfish.constants import Indices, IntensityIndices


class IntensityTable(xr.DataArray):

    # todo coords accepts super complicated types
    def __init__(self, intensities, coords=None, dims: Tuple[str, ...]=None,
                 name: Union[str, Tuple[str]]=None, attrs: Dict=None, encoding: Dict=None, fastpath=False) -> None:
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
        super().__init__(
            data=intensities, coords=coords, dims=dims, name=name, attrs=attrs, encoding=encoding, fastpath=fastpath)

    @classmethod
    def from_spot_data(cls, intensity_data: pd.DataFrame, tile_data: pd.DataFrame, feature_attributes: pd.DataFrame):

        coords = (
            dataframe_to_multiindex(tile_data[[Indices.CH, Indices.HYB]]),
            dataframe_to_multiindex(feature_attributes)
        )

        intensity_table = cls(
            intensities=intensity_data.values,
            coords=coords,
            dims=(IntensityIndices.TILES.value, IntensityIndices.FEATURES.value)
        )

        return intensity_table.unstack(IntensityIndices.TILES.value)

    def save(self, filename) -> None:
        """Save an IntensityTable as a Netcdf File

        Parameters
        ----------
        filename : str
            Name of Netcdf file

        """
        # TODO when https://github.com/pydata/xarray/issues/1077 (support for multiindex serliazation) is merged, remove
        # this reset_index() call and simplify load, below
        self.reset_index('features').to_netcdf(filename)

    @classmethod
    def load(cls, filename):
        """load an IntensityTable from Netcdf

        Parameters
        ----------
        filename : str
            File to load

        """
        loaded = xr.open_dataarray(filename)
        intensity_table = cls(
            loaded.data,
            loaded.coords,
            loaded.dims
        )
        return intensity_table.set_index(features=list(intensity_table['features'].coords.keys()))

    def show(self, background_image: np.ndarray) -> None:
        """show spots on a background image"""
        raise NotImplementedError
