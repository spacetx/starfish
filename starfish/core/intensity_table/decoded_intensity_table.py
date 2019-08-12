import numpy as np
import pandas as pd

from starfish.core.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import (
    Axes,
    Coordinates,
    DecodedSpots,
    Features,
    SpotAttributes
)
from starfish.core.util.dtype import preserve_float_range


class DecodedIntensityTable(IntensityTable):
    """
    DecodedIntensityTable is a container for spot or pixel features extracted from
    image data that have been decoded. It is the primary output from starfish
    :py:class:`Decode` methods.

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
    ) -> "DecodedIntensityTable":
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

        return DecodedIntensityTable(intensities)

    @staticmethod
    def assign_synthetic_targets(intensities: IntensityTable) -> "DecodedIntensityTable":
        intensities = DecodedIntensityTable(intensities)
        intensities[Features.TARGET] = (Features.AXIS, np.random.choice(list('ABCD'), size=20))
        intensities[Features.DISTANCE] = (Features.AXIS, np.random.rand(20))
        return intensities

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
                Axes.X.value, Axes.Y.value, Axes.ZPLANE.value, Coordinates.X.value,
                Coordinates.Y.value, Coordinates.Z.value]]
        else:
            grouped = self.to_features_dataframe().groupby([Features.CELL_ID])[[
                Axes.X.value, Axes.Y.value, Axes.ZPLANE.value]]
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