from typing import Callable, Optional
import ray
import pandas as pd
import numpy as np
from copy import deepcopy

from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import SpotFindingResults
from starfish.types import Axes, Features
from ._base import DecodeSpotsAlgorithm


from .check_all_funcs import findNeighbors, buildBarcodes, decoder, distanceFilter, cleanup, removeUsedSpots
from .util import _merge_spots_by_round


class CheckAll(DecodeSpotsAlgorithm):
    """
    Decode spots by selecting the max-valued channel in each sequencing round.

    Note that this assumes that the codebook contains only one "on" channel per sequencing round,
    a common pattern in experiments that assign one fluorophore to each DNA nucleotide and
    read DNA sequentially. It is also a characteristic of single-molecule FISH and RNAscope
    codebooks.

    Parameters
    ----------
    codebook : Codebook
        Contains codes to decode IntensityTable
    trace_building_strategy: TraceBuildingStrategies
        Defines the strategy for building spot traces to decode across rounds and chs of spot
        finding results.
    search_radius : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        Number of pixels over which to search for spots in other rounds and channels.
    anchor_round : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    """

    def __init__(
            self,
            codebook: Codebook,
            filter_rounds: Optional[int]=None,
            search_radius: Optional[float]=3,
            round_omit_num: Optional[int]=0):
        self.codebook = codebook
        self.filterRounds = filter_rounds
        self.searchRadius = search_radius
        self.roundOmitNum = round_omit_num

    def run(self, spots: SpotFindingResults, n_processes: int=1, *args) -> DecodedIntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        spots: SpotFindingResults
            A Dict of tile indices and their corresponding measured spots

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """

        # Rename n_processes (trying to stay consistent between starFISH's _ variables and my camel case ones)
        numJobs = n_processes

        # If using an search radius exactly equal to a possible distance between two pixels (ex: 1), some 
        # distances will be calculated as slightly less than their exact distance (either due to rounding or
        # precision) so search radius needs to be slightly increased to ensure this doesn't happen
        self.searchRadius += 0.001

        # Initialize ray for multi_processing
        ray.init(num_cpus=numJobs)
        
        # Create dictionary where keys are round labels and the values are pandas dataframes containing information on
        # the spots found in that round
        spotTables = _merge_spots_by_round(spots)
        
        # If user did not specify the filterRounds variable (it will have default value -1) change it to either one less
        # than the number of rounds if roundOmitNum is 0 or the number of rounds minus the roundOmitNum if roundOmitNum > 0
        if self.filterRounds == None:
            if self.roundOmitNum == 0:
                self.filterRounds = len(spotTables) - 1
            else:
                self.filterRounds = len(spotTables) - self.roundOmitNum
        

        # Create dictionary of neighbors (within the search radius) in other rounds for each spot
        neighborDict = findNeighbors(spotTables, self.searchRadius)
        
        # Create dictionary with mapping from spot id in spotTables to channel number and one with spot
        # coordinates for fast access
        channelDict = {}
        spotCoords = {}
        for r in [*spotTables]:
            channelDict[r] = spotTables[r]['c'].to_dict()
            spotCoords[r] = spotTables[r][['z','y','x']].T.to_dict()            
        
        # Set list of round omission numbers to loop through
        roundOmits = range(self.roundOmitNum+1)
        
        # Decode for each round omission number 
        allCodes = pd.DataFrame()
        for currentRoundOmitNum in roundOmits:
            decodedTables = {}
            for r in range(len(spotTables)):
                roundData = deepcopy(spotTables[r])
                
                # Create dictionary of dataframes (based on perRoundSpotTables data) that contains additional columns for each spot
                # containing all the possible barcodes that could be constructed from the neighbors of that spot
                roundData = buildBarcodes(roundData, neighborDict, currentRoundOmitNum, channelDict, r, numJobs)
                
                # Match possible barcodes to codebook and add new columns with info about barcodes that had a codebook match
                roundData = decoder(roundData, self.codebook, currentRoundOmitNum, r, numJobs)

                # Choose most likely barcode for each spot in each round by find the possible decodable barcode with the least
                # spatial variance between the spots that made up the barcode
                roundData = distanceFilter(roundData, spotCoords, r, numJobs)
                
                # Assign to DecodedTables dictionary
                decodedTables[r] = roundData

            # Turn spot table dictionary into single table, filter barcodes by round frequency, add additional information,
            # and choose between barcodes that use the same spot(s)
            finalCodes = cleanup(decodedTables, spotCoords, self.filterRounds)
            
            # If this is not the last round omission number to run, remove spots that have just been found to be in
            # passing barcodes from neighborDict so they are not used for the next round omission number
            if currentRoundOmitNum != roundOmits[-1]:
                neighborDict = removeUsedSpots(finalCodes, neighborDict)
            
            # Append found codes to allCodes table
            allCodes = allCodes.append(finalCodes).reset_index(drop=True)
        
        # Shutdown ray
        ray.shutdown()


        # Create and fill in intensity table
        channels=spots.ch_labels
        rounds=spots.round_labels    

        # create empty IntensityTable filled with np.nan
        data = np.full((len(allCodes), len(channels), len(rounds)), fill_value=np.nan)
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
        centers = allCodes['center']
        coords: Mapping[Hashable, Tuple[str, Any]] = {
            Features.SPOT_RADIUS: (Features.AXIS, np.full(len(allCodes), 1)),
            Axes.ZPLANE.value: (Features.AXIS, np.asarray([round(c[2]) for c in centers])),
            Axes.Y.value: (Features.AXIS, np.asarray([round(c[1]) for c in centers])),
            Axes.X.value: (Features.AXIS, np.asarray([round(c[0]) for c in centers])),
            Features.SPOT_ID: (Features.AXIS, np.arange(len(allCodes))),
            Features.AXIS: (Features.AXIS, np.arange(len(allCodes))),
            Axes.ROUND.value: (Axes.ROUND.value, rounds),
            Axes.CH.value: (Axes.CH.value, channels)
        }
        intensity_table = IntensityTable(data=data, dims=dims, coords=coords)

        # Fill in data values
        table_codes = []
        for i in range(len(allCodes)):
            code = []
            for ch in allCodes.loc[i, 'best_barcodes']:
                # If a round is not used, row will be all zeros
                code.append(np.asarray([0 if j != ch else 1 for j in range(len(channels))]))
            table_codes.append(np.asarray(code).T)
        intensity_table.values = np.asarray(table_codes)
        intensity_table = transfer_physical_coords_to_intensity_table(intensity_table=intensity_table, spots=spots)
        intensities = intensity_table.transpose('features', 'r', 'c')

        self.codebook._validate_decode_intensity_input_matches_codebook_shape(intensities)

        # Create DecodedIntensityTable
        result=DecodedIntensityTable.from_intensity_table(
            intensities,
            targets=(Features.AXIS, allCodes['best_targets'].astype('U')),
            distances=(Features.AXIS, allCodes["best_distances"]),
            passes_threshold=(Features.AXIS, np.full(len(allCodes), True)),
            filter_tally=(Features.AXIS, allCodes['rounds_used']))


        return result
