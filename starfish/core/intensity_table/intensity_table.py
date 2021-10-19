from itertools import product
from json import loads
from typing import cast, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from starfish.core.types import (
    Axes,
    Coordinates,
    Features,
    LOG,
    OverlapStrategy,
    SpotAttributes,
    STARFISH_EXTRAS_KEY
)
from starfish.core.util.levels import preserve_float_range
from .overlap import (
    find_overlaps_of_xarrays,
    OVERLAP_STRATEGY_MAP,
)

CoordinatesType = Dict[Hashable, Union[Tuple[str, np.ndarray], np.ndarray]]


class IntensityTable(xr.DataArray):
    """
    IntensityTable is a container for spot or pixel features extracted from
    image data. It is the primary output from starfish :py:class:`SpotFinder`
    methods.

    An IntensityTable records the numeric intensity of a set of features in each
    :code:`(round, channel)` tile in which the feature is identified.
    The :py:class:`IntensityTable` has shape
    :code:`(n_feature, n_round, n_channel)`.

    Some :py:class:`SpotFinder` methods identify a position and search for
    Gaussian blobs in a small radius, only recording intensities if they are
    found in a given tile. Other :py:class:SpotFinder: approaches
    find blobs in a max-projection and measure them everywhere. As a result,
    some IntensityTables will be dense, and others will contain :code:`np.nan`
    entries where no feature was detected.

    Examples
    --------
    Create an IntensityTable using the ``SyntheticData`` factory::

        >>> from starfish.core.test.factories import SyntheticData
        >>> sd = SyntheticData(n_ch=3, n_round=4, n_codes=2, n_spots=3)
        >>> codes = sd.codebook()
        >>> sd.intensities(codebook=codes)
        <xarray.IntensityTable (features: 3, r: 4, c: 3)>
        array([[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]],

               [[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]],

               [[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]]], dtype=float32)
        Coordinates:
            z         (features) int64 8 6 2
            y         (features) float64 14.61 41.9 3.935
            x         (features) float64 11.0 42.42 5.249
            radius    (features) float64 nan nan nan
          * features  (features) int64 0 1 2
          * r         (r) int64 0 1 2 3
          * c         (c) int64 0 1 2
    """

    __slots__ = ()

    @staticmethod
    def _build_xarray_coords(
            spot_attributes: SpotAttributes,
            round_values: Sequence[int],
            channel_values: Sequence[int],
    ) -> CoordinatesType:
        """build coordinates for intensity-table"""
        coordinates: CoordinatesType = {
            k: (Features.AXIS, spot_attributes.data[k].values)
            for k in spot_attributes.data
        }
        coordinates.update({
            Features.AXIS: np.arange(len(spot_attributes.data)),
            Axes.ROUND.value: np.array(round_values),
            Axes.CH.value: np.array(channel_values),
        })
        return coordinates

    @classmethod
    def zeros(
            cls,
            spot_attributes: SpotAttributes,
            round_labels: Sequence[int],
            ch_labels: Sequence[int],
    ) -> "IntensityTable":
        """
        Create an empty intensity table with pre-set shape whose values are zero.

        Parameters
        ----------
        spot_attributes : SpotAttributes
            Table containing spot metadata. Must contain the values specified in Axes.X,
            Y, Z, and RADIUS.
        round_labels : Sequence[int]
            The possible values for the round number, in the order that they are in the ImageStack
            5D tensor.
        ch_labels : Sequence[int]
            The possible values for the channel number, in the order that they are in the ImageStack
            5D tensor.

        Returns
        -------
        IntensityTable :
            IntensityTable filled with zeros.

        """
        if not isinstance(spot_attributes, SpotAttributes):
            raise TypeError('parameter spot_attributes must be a starfish SpotAttributes object.')

        data = np.zeros((spot_attributes.data.shape[0], len(round_labels), len(ch_labels)))
        dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value,)
        coords = cls._build_xarray_coords(spot_attributes, round_labels, ch_labels)

        intensity_table = cls(
            data=data, coords=coords, dims=dims,
        )

        return intensity_table

    @classmethod
    def from_spot_data(
            cls,
            intensities: np.ndarray,
            spot_attributes: SpotAttributes,
            round_values: Sequence[int],
            ch_values: Sequence[int],
            *args, **kwargs) -> "IntensityTable":
        """
        Creates an IntensityTable from a :code:`(features, channel, round)`
        array and a :py:class:`~starfish.types._spot_attributes.SpotAttributes`
        object.

        Parameters
        ----------
        intensities : np.ndarray
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

        if len(round_values) != intensities.shape[1]:
            raise ValueError(
                f"The number of round values ({len(round_values)}) should be equal to intensities' "
                f"shape[1] ({intensities.shape[1]})."
            )

        if len(ch_values) != intensities.shape[2]:
            raise ValueError(
                f"The number of ch values ({len(ch_values)}) should be equal to intensities' "
                f"shape[2] ({intensities.shape[2]})."
            )

        if not isinstance(spot_attributes, SpotAttributes):
            raise TypeError('parameter spot_attributes must be a starfish SpotAttributes object.')

        coords = cls._build_xarray_coords(spot_attributes, round_values, ch_values)
        dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value)

        return cls(intensities, coords, dims, *args, **kwargs)

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

    def to_netcdf(self, filename: str) -> None:  # type: ignore
        """
        Save an IntensityTable as a Netcdf File.

        Parameters
        ----------
        filename : str
            Name of Netcdf file.

        """
        super().to_netcdf(filename)

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
        return intensities

    @classmethod
    def from_image_stack(
            cls,
            image_stack,
            crop_x: int = 0, crop_y: int = 0, crop_z: int = 0
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
            Axes.ROUND.value,
            Axes.CH.value,
        )

        # (pixels, ch, round)
        intensity_data = data.values.reshape(-1, image_stack.num_rounds, image_stack.num_chs)

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
            image_stack.axis_labels(Axes.ROUND),
            image_stack.axis_labels(Axes.CH),
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
        return cast(
            IntensityTable,
            xr.concat(intensity_tables, dim=Features.AXIS)
        )

    def to_features_dataframe(self) -> pd.DataFrame:
        """
        Generates a dataframe of the underlying features metadata.
        This is guaranteed to contain the features x, y, z, and radius.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(dict(self[Features.AXIS].coords))
