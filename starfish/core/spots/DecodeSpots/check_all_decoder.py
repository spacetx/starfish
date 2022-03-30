import sys
from collections import Counter
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
from .check_all_funcs import buildBarcodes, cleanup, createNeighborDict, createRefDicts, decoder, \
    distanceFilter, findNeighbors, removeUsedSpots
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
    5. Only keep barcodes/targets that were found as "best" using at least 2 of the spots that make
    each up
    6. Find maximum independent set (approximation) of the spot combinations so no two barcodes use
    the same spot
    7. Remove all spots used in decoded targets that passed the previous filtering steps from the
    original set of spots
    8. Rerun steps 2-5 for barcodes that use less than the full set of rounds for codebook
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
            error_rounds: int=0,
            mode='med',
            physical_coords=False):
        self.codebook = codebook
        self.searchRadius = search_radius
        self.errorRounds = error_rounds
        self.mode = mode
        self.physicalCoords = physical_coords

        # Error checking for some inputs

        # Check that codebook is the right class and not empty
        if not isinstance(self.codebook, Codebook) or len(codebook) == 0:
            sys.exit('codebook is either not a Codebook object or is empty')
        # Check that error_rounds is either 0 or 1
        if self.errorRounds not in [0, 1]:
            exit('error_rounds can only take a value of 0 or 1')
        # Return error if search radius is greater than 4.5 or negative
        if self.searchRadius < 0 or self.searchRadius > 4.5:
            sys.exit('search_radius must be positive w/ max value of 4.5')

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
        # Check that numJobs is a positive integer
        if numJobs < 0 or not isinstance(numJobs, int):
            sys.exit('n_process must be a positive integer')

        # Initialize ray for multi_processing
        ray.init(num_cpus=numJobs, ignore_reinit_error=True)

        # Create dictionary where keys are round labels and the values are pandas dataframes
        # containing information on the spots found in that round
        spotTables = _merge_spots_by_round(spots)

        # Check that enough rounds have spots to make at least one barcode
        spotsPerRound = [len(spotTables[r]) for r in range(len(spotTables))]
        counter = Counter(spotsPerRound)
        if counter[0] > self.errorRounds:
            exit('Not enough spots to form a barcode')

        if self.physicalCoords:
            physicalCoords = spots.physical_coord_ranges
            if len(physicalCoords['z'].data) > 1:
                zScale = physicalCoords['z'][1].data - physicalCoords['z'][0].data
            else:
                zScale = 1
            yScale = physicalCoords['y'][1].data - physicalCoords['y'][0].data
            xScale = physicalCoords['x'][1].data - physicalCoords['x'][0].data
            if xScale <= 0 or yScale <= 0 or zScale <= 0:
                exit('invalid physical coords')

        # Add one to channels labels (prevents collisions between hashes of barcodes later), adds
        # unique spot_id column for each spot in each round, and scales the x, y, and z columns to
        # the phsyical coordinates if specified
        for r in spots.round_labels:
            spotTables[r]['c'] += 1
            spotTables[r]['spot_id'] = range(1, len(spotTables[r]) + 1)
            if self.physicalCoords:
                spotTables[r]['z'] = spotTables[r]['z'] * zScale
                spotTables[r]['y'] = spotTables[r]['y'] * yScale
                spotTables[r]['x'] = spotTables[r]['x'] * xScale

        # Choose search radius set based on search_radius parameter and ability for spots to be
        # neighbors across z slices. Each value in allSearchRadii represents an incremental
        # increase in neighborhood size
        set1 = False
        zs = set()
        [zs.update(spotTables[r]['z']) for r in range(len(spotTables))]
        if self.physicalCoords:
            if zScale < self.searchRadius or len(zs) > 1:
                set1 = True
        else:
            if len(zs) > 1:
                set1 = True
        if set1:
            allSearchRadii = np.array([0, 1.05, 1.5, 1.8, 2.05, 2.3, 2.45, 2.85, 3.05, 3.2,
                                       3.35, 3.5, 3.65, 3.75, 4.05, 4.15, 4.25, 4.4, 4.5])
        else:
            allSearchRadii = np.array([0, 1.05, 1.5, 2.05, 2.3, 2.85, 3.05, 3.2, 3.65, 4.05, 4.15,
                                       4.25, 4.5])

        maxRadii = allSearchRadii[(allSearchRadii - self.searchRadius) <= 0][-1]
        radiusSet = allSearchRadii[allSearchRadii <= maxRadii]

        # Calculate neighbors for each radius in the set
        neighborsByRadius = {}
        for searchRadius in radiusSet:
            if self.physicalCoords:
                searchRadius = round(searchRadius * xScale, 5)
            neighborsByRadius[searchRadius] = findNeighbors(spotTables, searchRadius, numJobs)

        # Create reference dictionaries for spot channels, coordinates, raw intensities, and
        # normalized intensities. Each is a dict w/ keys equal to the round labels and each
        # value is a dict with spot IDs in that round as keys and their corresponding value
        # (channel label, spatial coords, etc)
        channelDict, spotCoords, spotIntensities, spotQualDict = createRefDicts(spotTables, numJobs)

        # Add spot quality (normalized spot intensity) tp spotTables
        for r in range(len(spotTables)):
            spotTables[r]['spot_quals'] = [spotQualDict[r][spot] for spot in
                                           spotTables[r]['spot_id']]

        # Set list of round omission numbers to loop through
        roundOmits = range(self.errorRounds + 1)

        # Set parameters according to presets
        if self.mode == 'high':
            strictnesses = [50, -1]
            seedNumbers = [len(spotTables) - 1, len(spotTables)]
            minDist = 3
            if self.errorRounds == 1:
                strictnesses.append(1)
                seedNumbers.append(len(spotTables) - 1)
        elif self.mode == 'med':
            strictnesses = [50, -5]
            seedNumbers = [len(spotTables) - 1, len(spotTables)]
            minDist = 3
            if self.errorRounds == 1:
                strictnesses.append(5)
                seedNumbers.append(len(spotTables) - 1)
        elif self.mode == 'low':
            strictnesses = [50, -100]
            seedNumbers = [len(spotTables) - 1, len(spotTables) - 1]
            minDist = 100
            if self.errorRounds == 1:
                strictnesses.append(10)
                seedNumbers.append(len(spotTables) - 1)
        else:
            exit('Invalid mode choice ("high", "med", or "low")')

        # Decode for each round omission number, store results in allCodes table
        allCodes = pd.DataFrame()
        for s, strictness in enumerate(strictnesses):
            seedNumber = seedNumbers[s]
            for currentRoundOmitNum in roundOmits:
                for intVal in range(50, -1, -50):

                    spotsPerRound = [len(spotTables[r]) for r in range(len(spotTables))]
                    counter = Counter(spotsPerRound)
                    condition3 = True if counter[0] > currentRoundOmitNum else False

                    if not condition3:
                        # Subset spots by intensity, start with top 50% then decode again with all
                        currentTables = {}
                        for r in range(len(spotTables)):
                            lowerBound = np.percentile(spotTables[r]['spot_quals'], intVal)
                            currentTables[r] = spotTables[r][spotTables[r]['spot_quals']
                                                             >= lowerBound]

                    # Decode each radius and remove spots found in each decoding before the next
                    for sr, searchRadius in enumerate(radiusSet):
                        if self.physicalCoords:
                            searchRadius = round(searchRadius * xScale, 5)

                        # Only run partial codes for the final strictness and don't run full
                        # barcodes for the final strictness. Also don't run if there are not
                        # enough spots left.
                        condition1 = (currentRoundOmitNum == 1 and s != len(strictnesses) - 1)
                        condition2 = (len(roundOmits) > 1 and currentRoundOmitNum == 0
                                      and s == len(strictnesses) - 1)

                        if condition1 or condition2 or condition3:
                            pass
                        else:

                            # Creates neighbor dictionary for the current radius and current set of
                            # spots
                            neighborDict = createNeighborDict(currentTables, searchRadius,
                                                              neighborsByRadius)

                            # Find best spot combination using each spot in each round as seed
                            decodedTables = {}
                            for r in range(len(spotTables)):

                                # roundData will carry the possible barcode info for each spot in
                                # the current round being examined
                                roundData = deepcopy(currentTables[r])
                                roundData = roundData.drop(['intensity', 'z', 'y', 'x', 'radius',
                                                            'c', 'spot_quals'], axis=1)

                                # From each spot's neighbors, create all possible combinations that
                                # would form a barocde with the correct number of rounds. Adds
                                # spot_codes column to roundData
                                roundData = buildBarcodes(roundData, neighborDict,
                                                          currentRoundOmitNum, channelDict,
                                                          strictness, r, numJobs)

                                # When strictness is positive, distanceFilter is run first on all
                                # the potential barcodes to choose the one with the minimum score
                                # (based on spatial variance of the spots and their intensities)
                                # which are then matched to the codebook. Spots that have more
                                # possible barcodes to choose between than the current strictness
                                # number are dropped as ambiguous. If strictness is negative, all
                                # the possible barcodes are instead first matched to the codebook
                                # and then the lowest scoring decodable spot combination is chosen
                                # for each spot. Spots that have more decodable barcodes to choose
                                # from than the strictness value (absolute value) are dropped.
                                # Positive strictness method has lower false positive rate but
                                # finds fewer targets while the negative strictness method has
                                # higher false positive rates but finds more targets
                                if strictness > 0:

                                    # Choose most likely combination of spots for each seed spot
                                    # using their spatial variance and normalized intensity values.
                                    # Adds distance column to roundData
                                    roundData = distanceFilter(roundData, spotCoords, spotQualDict,
                                                               r, currentRoundOmitNum, numJobs)

                                    # Match possible barcodes to codebook. Adds target column to
                                    # roundData
                                    roundData = decoder(roundData, self.codebook, channelDict,
                                                        strictness, currentRoundOmitNum, r, numJobs)

                                else:

                                    # Match possible barcodes to codebook. Adds target column to
                                    # roundData
                                    roundData = decoder(roundData, self.codebook, channelDict,
                                                        strictness, currentRoundOmitNum, r, numJobs)

                                    # Choose most likely combination of spots for each seed spot
                                    # using their spatial variance and normalized intensity values.
                                    # Adds distance column to roundData
                                    roundData = distanceFilter(roundData, spotCoords, spotQualDict,
                                                               r, currentRoundOmitNum, numJobs)

                                # Assign to DecodedTables dictionary
                                decodedTables[r] = roundData

                            # Turn spot table dictionary into single table, filter barcodes by
                            # round frequency, add additional information, and choose between
                            # barcodes that have overlapping spots
                            finalCodes = cleanup(decodedTables, spotCoords, channelDict,
                                                 strictness, currentRoundOmitNum, seedNumber)

                            # Remove spots that have just been found to be in passing barcodes from
                            # neighborDict so they are not used for the next decoding round and
                            # filter codes whose distance value is above the minimum
                            if len(finalCodes) > 0:
                                finalCodes = finalCodes[finalCodes['distance'] <= minDist]
                                spotTables = removeUsedSpots(finalCodes, spotTables)
                                currentTables = removeUsedSpots(finalCodes, currentTables)

                            # Append found codes to allCodes table
                            allCodes = allCodes.append(finalCodes).reset_index(drop=True)

        # Shutdown ray
        ray.shutdown()

        # Create and fill in intensity table
        channels = spots.ch_labels
        rounds = spots.round_labels

        # create empty IntensityTable filled with np.nan
        data = np.full((len(allCodes), len(channels), len(rounds)), fill_value=np.nan)
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
        centers = allCodes['center']
        coords: Mapping[Hashable, Tuple[str, Any]] = {
            Features.SPOT_RADIUS: (Features.AXIS, np.full(len(allCodes), 1)),
            Axes.ZPLANE.value: (Features.AXIS, np.asarray([round(c[0]) for c in centers])),
            Axes.Y.value: (Features.AXIS, np.asarray([round(c[1]) for c in centers])),
            Axes.X.value: (Features.AXIS, np.asarray([round(c[2]) for c in centers])),
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
            # ints = allCodes.loc[i, 'intensities']
            for j, ch in enumerate(allCodes.loc[i, 'best_barcodes']):
                # If a round is not used, row will be all zeros
                code.append(np.asarray([0 if k != ch - 1 else 1 for k in range(len(channels))]))
            table_codes.append(np.asarray(code).T)
        int_table.values = np.asarray(table_codes)
        int_table = transfer_physical_coords_to_intensity_table(intensity_table=int_table,
                                                                spots=spots)
        intensities = int_table.transpose('features', 'r', 'c')

        # Validate results are correct shape
        self.codebook._validate_decode_intensity_input_matches_codebook_shape(intensities)

        # Create DecodedIntensityTable
        result = DecodedIntensityTable.from_intensity_table(
            intensities,
            targets=(Features.AXIS, allCodes['targets'].astype('U')),
            distances=(Features.AXIS, allCodes["distance"]),
            passes_threshold=(Features.AXIS, np.full(len(allCodes), True)),
            rounds_used=(Features.AXIS, allCodes['rounds_used']))

        return result
