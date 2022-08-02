from collections import Counter
from copy import deepcopy
from typing import Any, Hashable, Mapping, Tuple

import numpy as np
import pandas as pd

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
    Decode spots by generating all possible combinations of neighboring spots to form barcodes
    given a radius distance that spots may be from each other in order to form a barcode. Then
    chooses the best set of nonoverlapping spot combinations by choosing the ones with the least
    spatial variance of their spot coordinates, highest normalized intensity and are also found
    to be best for multiple spots in the barcode (see algorithm below). Allows for one error
    correction round (option for more may be added in the future).

    Two slightly different algorithms are used to balance the precision (proportion of targets that
    represent true mRNA molecules) and recall (proportion of true mRNA molecules that are
    recovered). They share mostly the same steps but two are switched between the different
    versions. The following is for the "filter-first" version:

    1. For each spot in each round, find all neighbors in other rounds that are within the search
    radius
    2. For each spot in each round, build all possible full length barcodes based on the channel
    labels of the spot's neighbors and itself
    3. Choose the "best" barcode of each spot's possible barcodes by calculating a score that is
    based on minimizing the spatial variance and maximizing the intensities of the spots in the
    barcode. Each spot is assigned a "best" barcode in this way.
    4. Drop "best" barcodes that don't have a matching target in the codebook
    5. Only keep barcodes/targets that were found as "best" using at least x of the spots that make
    each up (x is determined by parameters)
    6. Find maximum independent set (approximation) of the spot combinations so no two barcodes use
    the same spot

    The other method (which I'll call "decode-first") is the same except steps 3 and 4 are switched
    so that the minimum scoring barcode is chosen from the set of possible codes that have a match
    to the codebook. The filter-first method will return fewer decoded targets (lower recall) but
    has a lower false positive rate (higher precision) while the other method will find more targets
    (higher recall) but at the cost of an increased false positive rate (lower precision).

    Decoding is run in multiple stages with the parameters becoming less strict as it gets into
    later stages. The high accuracy algorithm (filter-first) is always run first followed by the low
    accuracy method (decode-first), each with slightly different parameters based on the choice of
    "mode" parameter. After each decoding, the spots found to be in decoded barcodes are removed
    from the original set of spots before they are decoded again with a new set of parameters. In
    order to simplify the number of parameters to choose from, I have sorted them into three sets of
    presets ("high", "medium", or "low" accuracy) determined by the "mode" parameter.

    Decoding is also done multiple times at multiple search radius values that start at 0 and
    increase incrementally until they reach the user-specified search radius. This allows high
    confidence barcodes to be called first and make things easier when later codes are called.

    If error_rounds is set to 1 (currently cannot handle more than 1), after running all decodings
    for barcodes that exactly match the codebook, another set of decodings will be run to find
    barcodes that are missing a spot in exactly one round. If the codes in the codebook all have a
    hamming distance of at least 2 from all other codes, each can still be uniquely identified
    using a partial code with a single round dropped. Barcodes decoded with a partial code like this
    are inherently less accurate and so an extra dimension called "rounds_used" was added to the
    DecodedIntensityTable output that labels each decoded target with the number of rounds that was
    used to decode it, allowing you to easily separate these less accurate codes from your high
    accuracy set if you wish


    Parameters
    ----------
    codebook : Codebook
        Contains codes to decode IntensityTable
    search_radius : float
        Maximum allowed distance (in pixels) that spots in different rounds can be from each other
        and still be allowed to be combined into a barcode together
    error_rounds : int
        Maximum hamming distance a barcode can be from it's target in the codebook and still be
        uniquely identified (i.e. number of error correction rounds in each the experiment)
    mode : string
        One of three preset parmaters sets. Choices are: "low", "med", or 'high'. Low accuracy mode
        will return more decoded targets but at the cost to accuracy (high recall, low precision)
        while the high accuracy version will find fewer false postives but also fewer targets
        overall (high precision, low recall), medium is a balance between the two.
    physical_coords : bool
        True or False, should decoding using physical distances from the original imagestack that
        you performed spot finding on? Should be used when distances between z pixels is much
        greater than distance between x and y pixels.
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
            raise ValueError(
                'codebook is either not a Codebook object or is empty')
        # Check that error_rounds is either 0 or 1
        if self.errorRounds not in [0, 1]:
            raise ValueError(
                'error_rounds can only take a value of 0 or 1')
        # Return error if search radius is greater than 4.5 or negative
        if self.searchRadius < 0 or self.searchRadius > 4.5:
            raise ValueError(
                'search_radius must be positive w/ max value of 4.5')

    def run(self,
            spots: SpotFindingResults,
            n_processes: int=1,
            *args) -> DecodedIntensityTable:
        """
        Decode spots by finding the set of nonoverlapping barcodes that have the minimum spatial
        variance within each barcode.
        Parameters
        ----------
        spots: SpotFindingResults
            A Dict of tile indices and their corresponding measured spots
        n_processes: int
            Number of threads to run decoder in parallel with
        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET values.
        """

        # Rename n_processes (trying to stay consistent between starFISH's _ variables and my
        # camel case ones)
        numJobs = n_processes
        # Check that numJobs is a positive integer
        if numJobs < 0 or not isinstance(numJobs, int):
            raise ValueError(
                'n_process must be a positive integer')

        # Create dictionary where keys are round labels and the values are pandas dataframes
        # containing information on the spots found in that round
        spotTables = _merge_spots_by_round(spots)

        # Check that enough rounds have spots to make at least one barcode
        spotsPerRound = [len(spotTables[r]) for r in range(len(spotTables))]
        counter = Counter(spotsPerRound)
        if counter[0] > self.errorRounds:
            raise ValueError(
                'Not enough spots to form a barcode')

        # If using physical coordinates, extract z and xy scales and check that they are all > 0
        if self.physicalCoords:
            physicalCoords = spots.physical_coord_ranges
            if len(physicalCoords['z'].data) > 1:
                zScale = physicalCoords['z'][1].data - physicalCoords['z'][0].data
            else:
                zScale = 1
            yScale = physicalCoords['y'][1].data - physicalCoords['y'][0].data
            xScale = physicalCoords['x'][1].data - physicalCoords['x'][0].data
            if xScale <= 0 or yScale <= 0 or zScale <= 0:
                raise ValueError(
                    'invalid physical coords')

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
        for r in range(len(spotTables)):
            zs.update(spotTables[r]['z'])
        if self.physicalCoords:
            if zScale < self.searchRadius and len(zs) > 1:
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

        maxRadii = allSearchRadii[(allSearchRadii - self.searchRadius) >= 0][0]
        radiusSet = allSearchRadii[allSearchRadii <= maxRadii]

        # Calculate neighbors for each radius in the set (done only once and referred back to
        # throughout decodings)
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

        # Set parameters according to presets (determined empirically). Strictness value determines
        # the decoding method used and the allowed number of possible barcode choices (positive
        # for filter-first, negative for decode-first).
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
            raise ValueError(
                'Invalid mode choice ("high", "med", or "low")')

        # Decode for each round omission number, intensity cutoff, and then search radius
        allCodes = pd.DataFrame()
        for currentRoundOmitNum in roundOmits:
            for s, strictness in enumerate(strictnesses):

                # Set seedNumber according to parameters for this strictness value
                seedNumber = seedNumbers[s]

                # First decodes only the highest normalized intensity spots then adds in the rest
                for intVal in range(50, -1, -50):

                    # First check that there are enough spots left otherwise an error will occur
                    spotsPerRound = [len(spotTables[r]) for r in range(len(spotTables))]
                    counter = Counter(spotsPerRound)
                    condition3 = True if counter[0] > currentRoundOmitNum else False
                    if not condition3:
                        # Subset spots by intensity, start with top 50% then decode again with all
                        currentTables = {}
                        for r in range(len(spotTables)):

                            if len(spotTables[r]) > 0:
                                lowerBound = np.percentile(spotTables[r]['spot_quals'], intVal)
                                currentTables[r] = spotTables[r][spotTables[r]['spot_quals']
                                                                 >= lowerBound]
                            else:
                                currentTables[r] = pd.DataFrame()

                    # Decode each radius and remove spots found in each decoding before the next
                    for sr, searchRadius in enumerate(radiusSet):

                        # Scale radius by xy scale if needed
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

                                if len(spotTables[r]) > 0:

                                    # roundData will carry the possible barcode info for each spot
                                    # in the current round being examined
                                    roundData = deepcopy(currentTables[r])

                                    # Drop all but the spot_id column
                                    roundData = roundData[['spot_id']]

                                    # From each spot's neighbors, create all possible combinations
                                    # that would form a barocde with the correct number of rounds.
                                    # Adds spot_codes column to roundData

                                    roundData = buildBarcodes(roundData, neighborDict,
                                                              currentRoundOmitNum, strictness, r,
                                                              numJobs)

                                    # When strictness is positive the filter-first methods is used
                                    # and distanceFilter is run first on all the potential barcodes
                                    # to choose the one with the minimum score (based on spatial
                                    # variance of the spots and their intensities) which are then
                                    # matched to the codebook. Spots that have more possible
                                    # barcodes to choose between than the current strictnessnumber
                                    # are dropped as ambiguous. If strictness is negative, the
                                    # decode-first method is run where all the possible barcodes
                                    # are instead first matched to the codebook and then the lowest
                                    # scoring decodable spot combination is chosen for each spot.
                                    # Spots that have more decodable barcodes to choose from than
                                    # the strictness value (absolute value) are dropped.
                                    if strictness > 0:

                                        # Choose most likely combination of spots for each seed
                                        # spot using their spatial variance and normalized intensity
                                        # values. Adds distance column to roundData
                                        roundData = distanceFilter(roundData, spotCoords,
                                                                   spotQualDict, r,
                                                                   currentRoundOmitNum, numJobs)

                                        # Match possible barcodes to codebook. Adds target column
                                        # to roundData
                                        roundData = decoder(roundData, self.codebook, channelDict,
                                                            strictness, currentRoundOmitNum, r,
                                                            numJobs)

                                    else:

                                        # Match possible barcodes to codebook. Adds target column
                                        # to roundData
                                        roundData = decoder(roundData, self.codebook, channelDict,
                                                            strictness, currentRoundOmitNum, r,
                                                            numJobs)

                                        # Choose most likely combination of spots for each seed
                                        # spot using their spatial variance and normalized
                                        # intensity values. Adds distance column to roundData
                                        roundData = distanceFilter(roundData, spotCoords,
                                                                   spotQualDict, r,
                                                                   currentRoundOmitNum, numJobs)

                                    # Assign to DecodedTables dictionary
                                    decodedTables[r] = roundData

                                else:
                                    decodedTables[r] = pd.DataFrame()

                            # Turn spot table dictionary into single table, filter barcodes by
                            # the seed number, add additional information, and choose between
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

        # Create and fill in intensity table
        channels = spots.ch_labels
        rounds = spots.round_labels

        # create empty IntensityTable filled with np.nan
        data = np.full((len(allCodes), len(channels), len(rounds)), fill_value=np.nan)
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)

        if len(allCodes) == 0:
            centers = []
        else:
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

        # If no targets found returns empty DecodedIntensityTable
        if len(allCodes) > 0:
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

            # Validate results are correct shape
            self.codebook._validate_decode_intensity_input_matches_codebook_shape(int_table)

            # Create DecodedIntensityTable
            result = DecodedIntensityTable.from_intensity_table(
                int_table,
                targets=(Features.AXIS, allCodes['targets'].astype('U')),
                distances=(Features.AXIS, allCodes["distance"]),
                passes_threshold=(Features.AXIS, np.full(len(allCodes), True)),
                rounds_used=(Features.AXIS, allCodes['rounds_used']))
        else:
            result = DecodedIntensityTable.from_intensity_table(
                int_table,
                targets=(Features.AXIS, np.array([])),
                distances=(Features.AXIS, np.array([])),
                passes_threshold=(Features.AXIS, np.array([])),
                rounds_used=(Features.AXIS, np.array([])))

        return result
