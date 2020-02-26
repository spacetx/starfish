from dataclasses import dataclass
from typing import Any, Hashable, Mapping, MutableMapping, Optional, Sequence, Tuple

import xarray as xr

from starfish.core.types import Axes, Coordinates, SpotAttributes
from starfish.core.util.logging import Log


AXES_ORDER = (Axes.ROUND, Axes.CH)


@dataclass
class PerImageSliceSpotResults:
    """
    Named tuple that gets returned in every spot finding's image_to_spots method, spot_attrs are the
    SpotAttributes and extras is any extra information collected from the spot finding process.
    """
    spot_attrs: SpotAttributes
    extras: Optional[Mapping[str, Any]]


class SpotFindingResults:
    """
    Wrapper class that describes the results from a spot finding method. The results
    mapping is a collection of (round, ch) indices and their corresponding measured
    SpotAttributes.
    """

    def __init__(
            self,
            imagestack_coords,
            log: Log,
            spot_attributes_list: Optional[
                Sequence[Tuple[PerImageSliceSpotResults, Mapping[Axes, int]]]] = None,
    ):
        """
        Construct a SpotFindingResults instance

        Parameters
        -----------
        imagestack_coords : xr.CoordinateArray
            The physical coordinate ranges of the ImageStack spots were found in
        log : Log
            The provenance log information from the ImageStack spots were found in.
        spot_attributes_list : Optional[
        Sequence[Tuple[PerImageSliceSpotResults, Mapping[Axes, int]]]]
            If spots were found using ImageStack.transform() the result is a list of
            tuples (PerImageSliceSpotResults, indices).  Indices should be a mapping from axes to
            axis value.  Instantiating SpotFindingResults with this list will convert the
            information to a dictionary.
        """
        spot_attributes_list = spot_attributes_list or []
        self._results: MutableMapping[Tuple, PerImageSliceSpotResults] = {
            tuple(indices[i] for i in AXES_ORDER): spots
            for spots, indices in spot_attributes_list
        }
        self.physical_coord_ranges: Mapping[Hashable, xr.DataArray] = {
            Axes.X.value: imagestack_coords[Coordinates.X.value],
            Axes.Y.value: imagestack_coords[Coordinates.Y.value],
            Axes.ZPLANE.value: imagestack_coords[Coordinates.Z.value]}
        self._log: Log = log

    def __setitem__(self, indices: Mapping[Axes, int], value: PerImageSliceSpotResults):
        """
        Add the round, ch indices and corresponding SpotAttributes to the results dict.

        Parameters
        ----------
        indices: Mapping[Axes, int]
            Mapping of Axes to int values
        spots: SpotAttributes
            Describes spots found on this tile.
        """
        round_ch_index = tuple(indices[i] for i in AXES_ORDER)
        self._results[round_ch_index] = value

    def __getitem__(self, indices: Mapping[Axes, int]) -> PerImageSliceSpotResults:
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

    def items(self):
        """
        Return iterator for (r,ch) and SpotAttributes in Spot finding results
        """
        return self._results.items()

    def keys(self):
        """
        Return all round, ch index pairs.
        """
        return self._results.keys()

    def values(self):
        """
        Return all SpotAttributes across rounds and chs.
        """
        return self._results.values()

    @property
    def round_labels(self):
        """
        Return the set of round labels in the SpotFindingResults
        """
        return sorted(set(r for (r, ch) in self.keys()))

    @property
    def ch_labels(self):
        """
        Return the set of ch labels in the SpotFindingResults
        """

        return sorted(set(ch for (r, ch) in self.keys()))

    def count_total_spots(self):
        """
        Return the total number of unique spots represented in the SpotFindingResults.
        """
        total = 0
        for spot_info in self.values():
            total += spot_info.spot_attrs.data.spot_id.size
        return total

    @property
    def get_physical_coord_ranges(self) -> Mapping[Hashable, xr.DataArray]:
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
        Log
        """
        return self._log
