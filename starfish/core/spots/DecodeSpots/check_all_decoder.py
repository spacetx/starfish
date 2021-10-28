from copy import deepcopy
from typing import Any, Hashable, Mapping, Tuple

import numpy as np
import pandas as pd
import ray

from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.types import SpotFindingResults
from starfish.types import Axes, Features
from ._base import DecodeSpotsAlgorithm
from .check_all_funcs import buildBarcodes, cleanup, createRefDicts, decoder, distanceFilter, \
    removeUsedSpots
from .util import _merge_spots_by_round


class CheckAll(DecodeSpotsAlgorithm):
    """
    Decode spots by generating all possible combinations of spots to form barcodes given a radius
    distance that spots must be from each other in order to form a barcode. Then chooses the best
    set of nonoverlapping spot combinations by choosing the ones with the least spatial variance
    of their spot coordinates and are also found to be best for multiple spots in the barcode
    (see algorithm below). Allows for error correction rounds.

    (see input parmeters below)
    1. For each spot in each round, find all neighbors in other rounds that are within the search
    radius
    2. For each spot in each round, build all possible full length barcodes based on the channel
    labels of the spot's neighbors and itself
    3. Drop barcodes that don't have a matching target in the codebook
    4. Choose the "best" barcode of each spot's possible target matching barcodes by calculating
    the sum of variances for each of the spatial coordinates of the spots that make up each barcode
    and choosing the minimum distance barcode (if there is a tie, they are all dropped as
    ambiguous). Each spot is assigned a "best" barcode in this way.
    5. Only keep barcodes/targets that were found as "best" in each of the rounds they have spots in
    (End here if number of error_rounds = 0)
    6. Remove all spots used in decoded targets that passed the previous filtering steps from the
    original set of spots
    7. Rerun steps 2-5 for barcodes that use less than the full set of rounds for codebook
    matching (how many rounds can be dropped determined by error_rounds parameter)

    Parameters
    ----------
    codebook : Codebook
        Contains codes to decode IntensityTable
    search_radius : float
        Number of pixels over which to search for spots in other rounds and channels.
    error_rounds : int
        Maximum hamming distance a barcode can be from it's target in the codebook and still be
        uniquely identified (i.e. number of error correction rounds in each the experiment)
    """

    def __init__(
            self,
            codebook: Codebook,
            search_radius: float=3,
            error_rounds: int=0):
        self.codebook = codebook
        self.searchRadius = search_radius
        self.errorRounds = error_rounds

    def run(self,
            spots: SpotFindingResults,
            n_processes: int=1,
            *args) -> DecodedIntensityTable:
        """
        Decode spots by finding the set of nonoverlapping barcodes that have the minimum spatial
        variance within each barcode

        Parameters
        ----------
        spots: SpotFindingResults
            A Dict of tile indices and their corresponding measured spots

        n_processes: int
            Number of threads to run decoder in parallel with

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """

        # Rename n_processes (trying to stay consistent between starFISH's _ variables and my
        # camel case ones)
        numJobs = n_processes

        # If using an search radius exactly equal to a possible distance between two pixels
        # (ex: 1), some distances will be calculated as slightly less than their exact distance
        # (either due to rounding or precision errors) so search radius needs to be slightly
        # increased to ensure this doesn't happen
        self.searchRadius += 0.001

        # Initialize ray for multi_processing
        ray.init(num_cpus=numJobs)

        # Create dictionary where keys are round labels and the values are pandas dataframes
        # containing information on the spots found in that round
        spotTables = _merge_spots_by_round(spots)

        # Add one to channels labels (prevents collisions between hashes of barcodes later)
        for r in spots.round_labels:
            spotTables[r]['c'] += 1

        # Set list of round omission numbers to loop through
        roundOmits = range(self.errorRounds + 1)

        # Decode for each round omission number, store results in allCodes table
        allCodes = pd.DataFrame()
        for currentRoundOmitNum in roundOmits:

            # Create necessary reference dictionaries
            neighborDict, channelDict, spotCoords = createRefDicts(spotTables, self.searchRadius)

            # Chooses best barcode for all spots in each round sequentially (possible barcode
            # space can become quite large which can increase memory needs so I do it this way so
            # we only need to store all potential barcodes that originate from one round at a
            # time)
            decodedTables = {}
            for r in range(len(spotTables)):
                roundData = deepcopy(spotTables[r])

                # Create dictionary of dataframes (based on spotTables data) that contains
                # additional columns for each spot containing all the possible barcodes that
                # could be constructed from the neighbors of that spot
                roundData = buildBarcodes(roundData, neighborDict, currentRoundOmitNum,
                                          channelDict, r, numJobs)

                # Match possible barcodes to codebook and add new columns with info about barcodes
                # that had a codebook match
                roundData = decoder(roundData, self.codebook, currentRoundOmitNum, r, numJobs)

                # Choose most likely barcode for each spot in each round by find the possible
                # decodable barcode with the least spatial variance between the spots that made up
                # the barcode
                roundData = distanceFilter(roundData, spotCoords, r, numJobs)

                # Assign to DecodedTables dictionary
                decodedTables[r] = roundData

            # Turn spot table dictionary into single table, filter barcodes by round frequency, add
            # additional information, and choose between barcodes that have overlapping spots
            finalCodes = cleanup(decodedTables, spotCoords, channelDict, currentRoundOmitNum)

            # If this is not the last round omission number to run, remove spots that have just
            # been found to be in passing barcodes from spotTables so they are not used for the
            # next round omission number
            if currentRoundOmitNum != roundOmits[-1]:
                spotTables = removeUsedSpots(finalCodes, spotTables)

            # Append found codes to allCodes table
            allCodes = allCodes.append(finalCodes).reset_index(drop=True)

        # Shutdown ray
        ray.shutdown()

        # Create and fill in intensity table
        channels = spots.ch_labels
        rounds = spots.round_labels

        # create empty IntensityTable filled with np.nan
        data = np.full((len(allCodes), len(rounds), len(channels)), fill_value=np.nan)
        dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value)
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
        int_table = IntensityTable(data=data, dims=dims, coords=coords)

        # Fill in data values
        table_codes = []
        for i in range(len(allCodes)):
            code = []
            for ch in allCodes.loc[i, 'best_barcodes']:
                # If a round is not used, row will be all zeros
                code.append(np.asarray([0 if j != ch else 1 for j in range(len(channels))]))
            table_codes.append(np.asarray(code))
        int_table.values = np.asarray(table_codes)
        int_table = transfer_physical_coords_to_intensity_table(intensity_table=int_table,
                                                                spots=spots)

        # Validate results are correct shape
        self.codebook._validate_decode_intensity_input_matches_codebook_shape(int_table)

        # Create DecodedIntensityTable
        result = DecodedIntensityTable.from_intensity_table(
            int_table,
            targets=(Features.AXIS, allCodes['best_targets'].astype('U')),
            distances=(Features.AXIS, allCodes["best_distances"]),
            passes_threshold=(Features.AXIS, np.full(len(allCodes), True)),
            rounds_used=(Features.AXIS, allCodes['rounds_used']))

        return result
