from typing import List, Mapping, Optional, Tuple

import xarray as xr

from starfish.core.types import Axes, Coordinates, SpotAttributes
from starfish.core.util.logging import Log


AXES_ORDER = (Axes.ROUND, Axes.CH)


class SpotFindingResults:
    """
    Wrapper class that describes the results from a spot finding method. The
    results dict is a collection of (round,ch indices) and their corresponding measured
    SpotAttributes.
    """

    def __init__(self, imagestack, spot_attributes_list: Optional[List[Tuple]] = None):
        """
        Construct a SpotFindingResults instance

        Parameters
        -----------
        spot_attributes_list : Optional[List[Tuple[indices, SpotAttributes]]]
            If spots were fond using Imagestack.transform() the result is a list of
            tuples (indices, SpotAttributes). Instantiating SpotFindingResults with
            this list will convert the information to a dictionary.
        """
        self._results: Mapping[Tuple, SpotAttributes] = dict()
        if spot_attributes_list:
            for indices, spots in spot_attributes_list:
                self._results[indices] = spots
        self.physical_coord_ranges: Mapping[str, xr.DataArray] = {
            Axes.X.value: imagestack.xarray[Coordinates.X.value],
            Axes.Y.value: imagestack.xarray[Coordinates.Y.value],
            Axes.ZPLANE.value: imagestack.xarray[Coordinates.Z.value]}
        self._log: Log = imagestack.log

    def set_spots_for_round_ch(self, indices: Mapping[Axes, int], spots: SpotAttributes
                               ) -> None:
        """
        Add the r,ch indices and corresponding SpotAttributes to the results dict.

        Parameters
        ----------
        indices: Mapping[Axes, int]
            Mapping of Axes to int values
        spots: SpotAttributes
            Describes spots found on this tile.
        """
        tile_index = tuple(indices[i] for i in AXES_ORDER)
        self._results[tile_index] = spots  # type: ignore

    def get_spots_for_round_ch(self, indices: Mapping[Axes, int]) -> SpotAttributes:
        """
        Returns the spots found in a given round and ch.

        Parameters
        ----------
        indices: Mapping[Axes, int]
            Mapping of Axes to int values

        Returns
        --------
        SpotAttributes
        """
        round_ch_index = tuple(indices[i] for i in AXES_ORDER)
        return self._results[round_ch_index]

    def round_ch_indices(self):
        """
        Return all round, ch index pairs.
        """
        return self._results.keys()

    def all_spots(self):
        """
        Return all list of SpotAttributes.
        """
        return self._results.values()

    @property
    def round_labels(self):
        """
        Return the set of Round labels in the SpotFindingResults
        """
        return list(set(sorted(r for (r, ch) in self.round_ch_indices())))

    @property
    def ch_labels(self):
        """
        Return the set of Ch labels in the SpotFindingResults
        """
        return list(set(sorted(ch for (c, ch) in self.round_ch_indices())))

    @property
    def get_physical_coord_ranges(self) -> Mapping[str, xr.DataArray]:
        """
        Returns the physical coordinate ranges the SpotResults cover. Needed
        information for calculating the physical coordinate values of decoded spots.
        """
        return self.physical_coord_ranges

    @property
    def log(self) -> Log:
        """
        Returns a list of pipeline components that have been applied to this get these SpotResults
        as well as their corresponding runtime parameters.

        For more information about provenance logging see
        `Provenance Logging
        <https://spacetx-starfish.readthedocs.io/en/latest/help_and_reference/api/utils/ilogging.html>`_

        Returns
        -------
        List[dict]
        """
        return self._log
