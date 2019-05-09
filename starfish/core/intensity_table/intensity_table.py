from itertools import product
from json import loads
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from starfish.core.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.core.types import (
    Axes,
    Coordinates,
    DecodedSpots,
    Features,
    LOG,
    OverlapStrategy,
    SpotAttributes,
    STARFISH_EXTRAS_KEY
)
from starfish.core.util.dtype import preserve_float_range
from .overlap import (
    find_overlaps_of_xarrays,
    OVERLAP_STRATEGY_MAP,
)


class IntensityTable(xr.DataArray):
    """
    IntensityTable is a container for spot or pixel features extracted from
    image data. It is the primary output from starfish :py:class:`SpotFinder`
    methods.

    An IntensityTable records the numeric intensity of a set of features in each
    :code:`(round, channel)` tile in which the feature is identified.
    The :py:class:`IntensityTable` has shape
    :code:`(n_feature, n_channel, n_round)`.

    Some :py:class:`SpotFinder` methods identify a position and search for
    Gaussian blobs in a small radius, only recording intensities if they are
    found in a given tile. Other :py:class:SpotFinder: approaches
    find blobs in a max-projection and measure them everywhere. As a result,
    some IntensityTables will be dense, and others will contain :code:`np.nan`
    entries where no feature was detected.

    Examples
    --------
    Create an IntensityTable using the ``synthetic_intensities`` method::

        >>> from starfish.core.test.factories import SyntheticData
        >>> sd = SyntheticData(n_ch=3, n_round=4, n_codes=2)
        >>> codes = sd.codebook()
        >>> sd.intensities(codebook=codes)
        <xarray.IntensityTable (features: 2, c: 3, h: 4)>
        array([[[    0.,     0.,     0.,     0.],
                [    0.,     0.,  8022., 12412.],
                [11160.,  9546.,     0.,     0.]],

               [[    0.,     0.,     0.,     0.],
                [    0.,     0., 10506., 10830.],
                [11172., 12331.,     0.,     0.]]])
        Coordinates:
        * features   (features) MultiIndex
        - z          (features) int64 7 3
        - y          (features) int64 14 32
        - x          (features) int64 32 15
        - r          (features) float64 nan nan
        * c          (c) int64 0 1 2
        * h          (h) int64 0 1 2 3
          target     (features) object 08b1a822-a1b4-4e06-81ea-8a4bd2b004a9 ...

    """

    @staticmethod
    def _build_xarray_coords(
            spot_attributes: SpotAttributes,
            channel_values: Sequence[int],
            round_values: Sequence[int]
    ) -> Dict[str, np.ndarray]:
        """build coordinates for intensity-table"""
        coordinates = {
            k: (Features.AXIS, spot_attributes.data[k].values)
            for k in spot_attributes.data}
        coordinates.update({
            Features.AXIS: np.arange(len(spot_attributes.data)),
            Axes.CH.value: np.array(channel_values),
            Axes.ROUND.value: np.array(round_values),
        })
        return coordinates

    @classmethod
    def zeros(
            cls,
            spot_attributes: SpotAttributes,
            ch_values: Sequence[int],
            round_values: Sequence[int],
    ) -> "IntensityTable":
        """
        Create an empty intensity table with pre-set shape whose values are zero.

        Parameters
        ----------
        spot_attributes : SpotAttributes
            Table containing spot metadata. Must contain the values specified in Axes.X,
            Y, Z, and RADIUS.
        ch_values : Sequence[int]
            The possible values for the channel number, in the order that they are in the ImageStack
            5D tensor.
        round_values : Sequence[int]
            The possible values for the round number, in the order that they are in the ImageStack
            5D tensor.

        Returns
        -------
        IntensityTable :
            IntensityTable filled with zeros.

        """
        if not isinstance(spot_attributes, SpotAttributes):
            raise TypeError('parameter spot_attributes must be a starfish SpotAttributes object.')

        data = np.zeros((spot_attributes.data.shape[0], len(ch_values), len(round_values)))
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
        coords = cls._build_xarray_coords(
            spot_attributes, np.array(ch_values), round_values)

        intensity_table = cls(
            data=data, coords=coords, dims=dims,
        )

        return intensity_table

    @classmethod
    def from_spot_data(
            cls,
            intensities: Union[xr.DataArray, np.ndarray],
            spot_attributes: SpotAttributes,
            ch_values: Sequence[int],
            round_values: Sequence[int],
            *args, **kwargs) -> "IntensityTable":
        """
        Creates an IntensityTable from a :code:`(features, channel, round)`
        array and a :py:class:`~starfish.types._spot_attributes.SpotAttributes`
        object.

        Parameters
        ----------
        intensities : Union[xr.DataArray, np.ndarray]
            Intensity data.
        spot_attributes : SpotAttributes
            Table containing spot metadata. Must contain the values specified in Axes.X,
            Y, Z, and RADIUS.
        ch_values : Sequence[int]
            The possible values for the channel number, in the order that they are in the ImageStack
            5D tensor.
        round_values : Sequence[int]
            The possible values for the round number, in the order that they are in the ImageStack
        args :
            Additional arguments to pass to the xarray constructor.
        kwargs :
            Additional keyword arguments to pass to the xarray constructor.

        Returns
        -------
        IntensityTable :
            IntensityTable containing data from passed intensities, annotated by spot_attributes
        """

        if len(intensities.shape) != 3:
            raise ValueError(
                f'intensities must be a (features * ch * round) 3-d tensor. Provided intensities '
                f'shape ({intensities.shape}) is invalid.')

        if len(ch_values) != intensities.shape[1]:
            raise ValueError(
                f"The number of ch values ({len(ch_values)}) should be equal to intensities' "
                f"shape[1] ({intensities.shape[1]})."
            )

        if len(round_values) != intensities.shape[2]:
            raise ValueError(
                f"The number of round values ({len(ch_values)}) should be equal to intensities' "
                f"shape[2] ({intensities.shape[2]})."
            )

        if not isinstance(spot_attributes, SpotAttributes):
            raise TypeError('parameter spot_attributes must be a starfish SpotAttributes object.')

        coords = cls._build_xarray_coords(spot_attributes, ch_values, round_values)

        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)

        intensities = cls(intensities, coords, dims, *args, **kwargs)
        return intensities

    def get_log(self):
        """
        Deserialize and return a list of pipeline components that have been applied
        throughout a starfish session to create this :py:class:IntensityTable:.
        """

        if STARFISH_EXTRAS_KEY in self.attrs and LOG in self.attrs[STARFISH_EXTRAS_KEY]:
            return loads(self.attrs[STARFISH_EXTRAS_KEY])[LOG]
        else:
            raise RuntimeError('No log info found.')

    @property
    def has_physical_coords(self):
        """Returns True if this table's features have physical-space loci."""
        return Coordinates.X in self.coords and Coordinates.Y in self.coords

    def to_netcdf(self, filename: str) -> None:
        """
        Save an IntensityTable as a Netcdf File.

        Parameters
        ----------
        filename : str
            Name of Netcdf file.

        """
        super().to_netcdf(filename)

    def to_mermaid(self, filename: str) -> pd.DataFrame:
        """
        Writes a .csv.gz file in columnar format that is readable by MERMAID visualization
        software.

        To run MERMAID, follow the installation instructions for that repository and simply
        replace the data.csv.gz file with the output of this function.

        Parameters
        ----------
        filename : str
            Name for compressed-gzipped MERMAID data file. Should end in '.csv.gz'.

        Notes
        ------
        See also https://github.com/JEFworks/MERmaid

        """

        # verify the IntensityTable has been decoded
        if Features.TARGET not in self.coords.keys():
            raise RuntimeError(
                'IntensityTable must be decoded before it can be converted to MERMAID input.'
            )

        # construct the MERMAID dataframe. As MERMAID adds support for non-categorical variables,
        # additional columns can be added here
        df = self.to_features_dataframe()
        column_order = [
            Axes.X,
            Axes.Y,
            Features.TARGET,
            Features.TARGET,  # added twice to support simultaneous coding
        ]
        mermaid_data = df[column_order]

        # write to disk
        mermaid_data.to_csv(filename, compression='gzip', index=False)

    @classmethod
    def open_netcdf(cls, filename: str) -> "IntensityTable":
        """
        Load an IntensityTable from Netcdf.

        Parameters
        ----------
        filename : str
            File to load.

        Returns
        -------
        IntensityTable

        """
        loaded = xr.open_dataarray(filename)
        intensity_table = cls(
            loaded.data,
            loaded.coords,
            loaded.dims,
            attrs=loaded.attrs,
        )
        return intensity_table

    @classmethod
    def synthetic_intensities(
            cls,
            codebook,
            num_z: int = 12,
            height: int = 50,
            width: int = 40,
            n_spots: int = 10,
            mean_fluor_per_spot: int = 200,
            mean_photons_per_fluor: int = 50,
    ) -> "IntensityTable":
        """
        Creates an IntensityTable with synthetic spots, that correspond to valid
        codes in a provided codebook.

        Parameters
        ----------
        codebook : Codebook
            Starfish codebook object.
        num_z : int
            Number of z-planes to use when localizing spots.
        height : int
            y dimension of each synthetic plane.
        width : int
            x dimension of each synthetic plane.
        n_spots : int
            Number of spots to generate.
        mean_fluor_per_spot : int
            Mean number of fluorophores per spot.
        mean_photons_per_fluor : int
            Mean number of photons per fluorophore.

        Returns
        -------
        IntensityTable

        """

        # TODO nsofroniew: right now there is no jitter on x-y positions of the spots
        z = np.random.randint(0, num_z, size=n_spots)
        y = np.random.uniform(0, height - 1, size=n_spots)
        x = np.random.uniform(0, width - 1, size=n_spots)

        r = np.empty(n_spots)
        r.fill(np.nan)  # radius is a function of the point-spread gaussian size
        spot_attributes = SpotAttributes(
            pd.DataFrame(
                {Axes.ZPLANE.value: z,
                 Axes.Y.value: y,
                 Axes.X.value: x,
                 Features.SPOT_RADIUS: r}
            )
        )

        # empty data tensor
        data = np.zeros(shape=(n_spots, *codebook.shape[1:]))

        targets = np.random.choice(
            codebook.coords[Features.TARGET], size=n_spots, replace=True)
        expected_bright_locations = np.where(codebook.loc[targets])

        # create a binary matrix where "on" spots are 1
        data[expected_bright_locations] = 1

        # add physical properties of fluorescence
        data *= np.random.poisson(mean_photons_per_fluor, size=data.shape)
        data *= np.random.poisson(mean_fluor_per_spot, size=data.shape)

        # convert data to float for consistency with starfish
        data = preserve_float_range(data)
        assert 0 < data.max() <= 1

        intensities = cls.from_spot_data(
            data, spot_attributes, np.arange(data.shape[1]), np.arange(data.shape[2]))
        intensities[Features.TARGET] = (Features.AXIS, targets)

        return intensities

    @classmethod
    def from_image_stack(
            cls,
            image_stack,
            crop_x: int=0, crop_y: int=0, crop_z: int=0
    ) -> "IntensityTable":
        """Generate an IntensityTable from all the pixels in the ImageStack

        Parameters
        ----------
        crop_x : int
            Number of pixels to crop from both top and bottom of x.
        crop_y : int
            Number of pixels to crop from both top and bottom of y.
        crop_z : int
            Number of pixels to crop from both top and bottom of z.
        image_stack : ImageStack
            ImageStack containing pixels to be treated as intensities.

        Returns
        -------
        IntensityTable :
            IntensityTable containing one intensity per pixel (across channels and rounds).

        """

        # verify the image is large enough to crop
        assert crop_z * 2 < image_stack.shape['z']
        assert crop_y * 2 < image_stack.shape['y']
        assert crop_x * 2 < image_stack.shape['x']

        zmin = image_stack.axis_labels(Axes.ZPLANE)[crop_z]
        ymin = crop_y
        xmin = crop_x
        zmax = image_stack.axis_labels(Axes.ZPLANE)[-crop_z - 1]
        ymax = image_stack.shape['y'] - crop_y
        xmax = image_stack.shape['x'] - crop_x
        cropped_stack = image_stack.sel({Axes.ZPLANE: (zmin, zmax),
                                         Axes.Y: (ymin, ymax),
                                         Axes.X: (xmin, xmax)})

        data = cropped_stack.xarray.transpose(
            Axes.ZPLANE.value,
            Axes.Y.value,
            Axes.X.value,
            Axes.CH.value,
            Axes.ROUND.value,
        )

        # (pixels, ch, round)
        intensity_data = data.values.reshape(
            -1, image_stack.num_chs, image_stack.num_rounds)

        # IntensityTable pixel coordinates
        z = image_stack.axis_labels(Axes.ZPLANE)
        y = np.arange(ymin, ymax)
        x = np.arange(xmin, xmax)

        feature_attribute_data = pd.DataFrame(
            data=np.array(list(product(z, y, x))),
            columns=['z', 'y', 'x']
        )
        feature_attribute_data[Features.SPOT_RADIUS] = np.full(
            feature_attribute_data.shape[0], fill_value=0.5
        )

        pixel_coordinates = SpotAttributes(feature_attribute_data)

        return IntensityTable.from_spot_data(
            intensity_data,
            pixel_coordinates,
            image_stack.axis_labels(Axes.CH),
            image_stack.axis_labels(Axes.ROUND),
        )

    @staticmethod
    def _process_overlaps(
        intensity_tables: List["IntensityTable"],
        overlap_strategy: OverlapStrategy
    ) -> List["IntensityTable"]:
        """
        Find the overlapping sections between IntensityTables and process them according
        to the given overlap strategy
        """
        overlap_pairs = find_overlaps_of_xarrays(intensity_tables)
        for indices in overlap_pairs:
            overlap_method = OVERLAP_STRATEGY_MAP[overlap_strategy]
            idx1, idx2 = indices
            # modify IntensityTables based on overlap strategy
            it1, it2 = overlap_method(intensity_tables[idx1], intensity_tables[idx2])
            # replace IntensityTables in list
            intensity_tables[idx1] = it1
            intensity_tables[idx2] = it2
        return intensity_tables

    @staticmethod
    def concatenate_intensity_tables(
        intensity_tables: List["IntensityTable"],
        overlap_strategy: Optional[OverlapStrategy] = None
    ) -> "IntensityTable":
        """
        Parameters
        ----------
        intensity_tables: List[IntensityTable]
            List of IntensityTables to be combined.
        overlap_strategy


        Returns
        -------

        """
        if overlap_strategy:
            intensity_tables = IntensityTable._process_overlaps(
                intensity_tables, overlap_strategy
            )
        return xr.concat(intensity_tables, dim=Features.AXIS)

    def to_features_dataframe(self) -> pd.DataFrame:
        """
        Generates a dataframe of the underlying features metadata.
        This is guaranteed to contain the features x, y, z, and radius.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(dict(self[Features.AXIS].coords))

    def to_decoded_spots(self) -> DecodedSpots:
        """
        Generates a dataframe containing decoded spot information. Guaranteed to contain physical
        spot coordinates (z, y, x) and gene target. Does not contain pixel coordinates.
        """
        if Features.TARGET not in self.coords.keys():
            raise RuntimeError(
                "Intensities must be decoded before a DecodedSpots table can be produced.")
        df = self.to_features_dataframe()
        pixel_coordinates = pd.Index([Axes.X, Axes.Y, Axes.ZPLANE])
        df = df.drop(pixel_coordinates.intersection(df.columns), axis=1).drop(Features.AXIS, axis=1)
        return DecodedSpots(df)

    def to_expression_matrix(self) -> ExpressionMatrix:
        """
        Generates a cell x gene count matrix where each cell is annotated with spatial metadata.

        Requires that spots in the IntensityTable have been assigned to cells.

        Returns
        -------
        ExpressionMatrix :
            cell x gene expression table
        """
        if Features.CELL_ID not in self.coords:
            raise KeyError(
                "IntensityTable must have 'cell_id' assignments for each cell before this function "
                "can be called. See starfish.spots.AssignTargets.Label.")
        grouped = self.to_features_dataframe().groupby([Features.CELL_ID, Features.TARGET])
        counts = grouped.count().iloc[:, 0].unstack().fillna(0)
        if self.has_physical_coords:
            grouped = self.to_features_dataframe().groupby([Features.CELL_ID])[[
                Axes.X, Axes.Y, Axes.ZPLANE, Coordinates.X, Coordinates.Y, Coordinates.Z]]
        else:
            grouped = self.to_features_dataframe().groupby([Features.CELL_ID])[[
                Axes.X, Axes.Y, Axes.ZPLANE]]
        min_ = grouped.min()
        max_ = grouped.max()
        coordinate_df = min_ + (max_ - min_) / 2
        metadata = {name: (Features.CELLS, data.values) for name, data in coordinate_df.items()}
        metadata[Features.AREA] = (Features.CELLS, np.full(counts.shape[0], fill_value=np.nan))
        # add genes to the metadata
        metadata.update({Features.GENES: counts.columns.values})
        metadata.update({Features.CELL_ID: (Features.CELLS, counts.index.values)})

        mat = ExpressionMatrix(
            data=counts.values,
            dims=(Features.CELLS, Features.GENES),
            coords=metadata,
            name='expression_matrix'
        )
        return mat
