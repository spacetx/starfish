from typing import Optional, Tuple

import numpy as np
import pandas as pd

from starfish.core.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import (
    Axes,
    Coordinates,
    DecodedSpots,
    Features,
)


class DecodedIntensityTable(IntensityTable):
    """
    DecodedIntensityTable is a container for spot or pixel features extracted from image data
    that have been decoded. It is the primary output from starfish :py:class:`Decode` methods.
    An IntensityTable records the numeric intensity of a set of features in each
    :code:`(round, channel)` tile in which the feature is identified.
    The :py:class:`IntensityTable` has shape
    :code:`(n_feature, n_channel, n_round)`.
    Some :py:class:`SpotFinder` methods identify a position and search for Gaussian blobs in a
    small radius, only recording intensities if they are found in a given tile. Other
    :py:class:SpotFinder: approaches find blobs in a max-projection and measure them everywhere.
    As a result, some IntensityTables will be dense, and others will contain :code:`np.nan`
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

    __slots__ = ()

    @classmethod
    def from_intensity_table(
            cls,
            intensities: IntensityTable,
            targets: Tuple[str, np.ndarray],
            distances: Optional[Tuple[str, np.ndarray]] = None,
            passes_threshold: Optional[Tuple[str, np.ndarray]] = None,
            rounds_used: Optional[Tuple[str, np.ndarray]] = None):
        """
        Assign target values to intensities.
        Parameters
        ----------
        intensities : IntensityTable
            intensity_table to assign target values to
        targets : Tuple[str, np.ndarray]
            Target values to assign
        distances : Optional[Tuple[str, np.ndarray]]
            Corresponding array of distances from target for each feature
        passes_threshold : Optional[Tuple[str, np.ndarray]]
            Corresponding array of boolean values indicating if each itensity passed
            given thresholds.
        rounds_used: Optional[Tuple[str, np.ndarray]]
            Corresponding array of integers indicated the number of rounds this
            decoded intensity was found in
        Returns
        -------
        DecodedIntensityTable
        """

        intensities = cls(intensities)
        intensities[Features.TARGET] = targets
        if distances:
            intensities[Features.DISTANCE] = distances
        if passes_threshold:
            intensities[Features.PASSES_THRESHOLDS] = passes_threshold
        if rounds_used:
            intensities['rounds_used'] = rounds_used
        return intensities

    def to_decoded_dataframe(self) -> DecodedSpots:
        """
        Generates a dataframe containing decoded spot information. Guaranteed to contain physical
        spot coordinates (z, y, x) and gene target. Does not contain pixel coordinates.
        """
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
        # rename unassigned spots
        counts.rename(index={'nan': 'unassigned'}, inplace=True)
        # remove and store 'nan' target counts
        nan_target_counts = np.zeros(counts.shape[0])
        if 'nan' in counts.columns:
            nan_target_counts = counts['nan'].values
            counts.drop(columns='nan', inplace=True)
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
        metadata["number_of_undecoded_spots"] = (Features.CELLS, nan_target_counts)
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
