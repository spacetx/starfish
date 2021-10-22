import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, MutableMapping, Optional, Sequence, Tuple

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

    def save(self, output_dir_name: str) -> None:
        """Save spot finding results to series of files.

        Parameters
        ----------
        output_dir_name: str
            Location to save all files.

        """
        json_data: Dict[str, Any] = {}

        os.chdir(os.path.dirname(output_dir_name))
        base_name = os.path.basename(output_dir_name)

        coords = {}
        for key in self.physical_coord_ranges.keys():
            path = "{}coords_{}.nc".format(base_name, key)
            coords[key] = path
            self.physical_coord_ranges[key].to_netcdf(path)
        json_data["physical_coord_ranges"] = coords

        path = "{}log.arr"
        json_data["log"] = {}
        json_data["log"]["path"] = path.format(base_name)
        with open(path.format(base_name), "w") as f:
            f.write(self.log.encode())

        spot_attrs = {}
        for key in self._results.keys():
            path = "{}spots_{}_{}.nc".format(base_name, key[0], key[1])
            spot_attrs["{}_{}".format(key[0], key[1])] = path
            self._results[key].spot_attrs.save(path)
        json_data["spot_attrs"] = spot_attrs

        save = json.dumps(json_data)
        with open("{}SpotFindingResults.json".format(base_name), "w") as f:
            f.write(save)

    @classmethod
    def load(cls, json_file: str):
        """Load serialized spot finding results.

        Parameters:
        -----------
        json_file: str
            json file to read

        Returns:
        --------
        SpotFindingResults:
            Object containing loaded results

        """
        fl = open(json_file)
        data = json.load(fl)
        os.chdir(os.path.dirname(json_file))

        with open(data["log"]["path"]) as f:
            txt = json.load(f)['log']
            txt = json.dumps(txt)
            log = Log.decode(txt)

        rename_axes = {
            'x': Coordinates.X.value,
            'y': Coordinates.Y.value,
            'z': Coordinates.Z.value
        }
        coords = {}
        for coord, path in data["physical_coord_ranges"].items():
            coords[rename_axes[coord]] = xr.load_dataarray(path)

        spot_attributes_list = []
        for key, path in data["spot_attrs"].items():
            zero = int(key.split("_")[0])
            one = int(key.split("_")[1])
            index = {AXES_ORDER[0]: zero, AXES_ORDER[1]: one}
            spots = SpotAttributes.load(path)
            spot_attributes_list.append((PerImageSliceSpotResults(spots, extras=None), index))

        return SpotFindingResults(
            imagestack_coords=coords,
            log=log,
            spot_attributes_list=spot_attributes_list
        )

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
