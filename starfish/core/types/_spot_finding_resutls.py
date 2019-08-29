from typing import Mapping, Optional, List, Tuple

from starfish.core.types import Axes, SpotAttributes


AXES_ORDER = (Axes.ROUND, Axes.CH)
AXES_ORDER_IS_VOLUME = (Axes.ROUND, Axes.CH, Axes.ZPLANE)


class SpotFindingResults:
    """
    Wrapper class that describes the results from a spot finding method. The
    results dict is a collection of tile indices and their corresponding measured
    SpotAttributes.
    """

    def __init__(self, spot_attributes_list: Optional[List[Tuple]] = None):
        """
        Construct a SpotFindingResults instance

        Parameters
        -----------
        spot_attributes_list : Optional[List[Tuple[indices, SpotAttributes]]]
            If spots were fond using Imagestack.transform() the result is a list of
            tuples (indices, SpotAttributes). Instantiating SpotFindingResults with
            this list will convert the information to a dictionary.
        """
        if spot_attributes_list:
            for indices, spots in spot_attributes_list:
                self._results[indices] = spots
        else:
            self._results: Mapping[Tuple, SpotAttributes] = dict()

    def set_tile_spots(self, indices: Mapping[Axes, int], spots: SpotAttributes
                       ) -> None:
        """
        Add the tile indices and corresponding SpotAttributes to the results dict.

        Parameters
        ----------
        indices: Mapping[Axes, int]
            Mapping of Axes to int values
        spots: SpotAttributes
            Describes spots found on this tile.
        """
        if Axes.ZPLANE in indices:
            tile_index = tuple(indices[i] for i in AXES_ORDER_IS_VOLUME)
        else:
            tile_index = tuple(indices[i] for i in AXES_ORDER)
        self._results[tile_index] = spots

    def get_tile_spots(self, indices: Mapping[Axes, int]) -> SpotAttributes:
        """
        Returns the spots found on a given tile.

        Parameters
        ----------
        indices: Mapping[Axes, int]
            Mapping of Axes to int values

        Returns
        --------
        SpotAttributes
        """
        if Axes.ZPLANE in indices:
            tile_index = tuple(indices[i] for i in AXES_ORDER_IS_VOLUME)
        else:
            tile_index = tuple(indices[i] for i in AXES_ORDER)
        return self._results[tile_index]

    def tile_indices(self):
        """
        Return all tile indices.
        """
        return self._results.keys()

    def all_spots(self):
        """
        Return all SpotAttributes
        """
        return self._results.values()

    def round_labels(self):
        """
        Return the set of Round labels in the SpotFindingResults
        """
        return list(set(sorted(tile_index[0] for tile_index in self.tile_indices())))

    def ch_labels(self):
        """
        Return the set of Ch labels in the SpotFindingResults
        """
        return list(set(sorted(tile_index[1] for tile_index in self.tile_indices())))

    def z_planes(self):
        """
        Return the set of z planes in the SpotFindingResults, if spots were found in 3d.
        If not return a value error.
        """
        try:
            return list(set(sorted(tile_index[2] for tile_index in self.tile_indices())))
        except IndexError:
            raise ValueError("These SpotResults do not contain z values.")
