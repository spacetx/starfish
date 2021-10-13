from collections import Counter
from scipy.spatial import cKDTree
from copy import deepcopy
from itertools import product, chain, permutations
from collections import defaultdict
import ray
import numpy as np
import pandas as pd
import warnings
from starfish.types import Axes
from starfish.core.codebook.codebook import Codebook
warnings.filterwarnings('ignore')

def findNeighbors(spotTables: dict, searchRadius: float) -> dict:

    '''
    Function that takes spatial information from the spot tables from each round and creates a
    dictionary that contains all the neighbors for each spot in other rounds that are within the
    search radius.

    Parameters
    ----------
        spotTables : dict
            Dictionary with round labels as keys and pandas dataframes containing spot information
            for its key round as values (result of _merge_spots_by_round function)

        searchRadius : float
            Distance that spots can be from each other and still form a barcode

    Returns
    -------
        dict: a dictionary with the following structure:
            {round: {
                spotID in round: {
                    neighborRound:
                        [list of spotIDs in neighborRound within searchRadius of spotID in round]
                    }
                }
            }
    '''

    # Create empty neighbor dictionary
    neighborDict = {}
    for r in spotTables:
        neighborDict[r] = {i: defaultdict(list, {r: [i]}) for i in range(len(spotTables[r]))}

    # For each pairing of rounds, find all mutual neighbors within the search radius for each spot
    # and assigns them in the neighborDict dictionary
    # Number assigned each spot in neighborDict is the index of it's original location in
    # spotTables and is used to track each spot uniquely throughout
    for i, r1 in enumerate(range((len(spotTables)))):
        tree = cKDTree(spotTables[r1][['z', 'y', 'x']])
        for r2 in list(range((len(spotTables))))[i + 1:]:
            allNeighbors = tree.query_ball_point(spotTables[r2][['z', 'y', 'x']], searchRadius)
            for j, neighbors in enumerate(allNeighbors):
                if neighbors != []:
                    for neighbor in neighbors:
                        neighborDict[r1][neighbor][r2].append(j)
                        neighborDict[r2][j][r1].append(neighbor)

    return neighborDict


def buildBarcodes(roundData: pd.DataFrame,
                  neighborDict: dict,
                  roundOmitNum: int,
                  channelDict: dict,
                  currentRound: int,
                  numJobs: int) -> pd.DataFrame:

    '''
    Function that adds to the current rounds spot table all the possible barcodes that could be
    formed using the neighbors of each spot, spots without enough neighbors to form a barcode
    # are dropped.

    Parameters
    ----------
        roundData : dict
            Spot data table for the current round

        neighborDict : dict
            Dictionary that contains all the neighbors for each spot in other rounds that are
            within the search radius

        roundOmitNum : int
            Maximum hamming distance a barcode can be from it's target in the codebook and still
            be uniquely identified (i.e. number of error correction rounds in each the experiment

        channelDict : dict
            Dictionary with mappings between spot IDs and their channel labels

        currentRound : int
            Current round to build barcodes for (same round that roundData is from)

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        pd.DataFrame : Copy of roundData with additional columns which list all possible barcodes
                       that could be made from each spot's neighbors

    '''

    @ray.remote
    def barcodeBuildFunc(data: pd.DataFrame,
                         channelDict: dict,
                         rang: tuple,
                         roundOmitNum: int,
                         roundNum: int) -> tuple:
        '''
        Subfunction to buildBarcodes that allows it to run in parallel chunks using ray

        Parameters
        ----------
            data : pd.DataFrame
                Spot table for the current round

            channelDict : dict
                Dictionary mapping spot IDs to their channels labels

            rang : tuple
                Range of indices to build barcodes for in the current data object

            roundOmitNum : int
                Maximum hamming distance a barcode can be from it's target in the codebook and
                still be uniquely identified (i.e. number of error correction rounds in each the
                experiment)

            roundNum : int
                Current round

        Returns
        -------
            tuple : First element is a list of the possible spot codes while the second element is
                    a list of the possible barcodes
        '''

        # Build barcodes from neighbors
        # spotCodes are the ordered spot IDs of the spots making up each barcode while barcodes are
        # the corresponding channel labels, need spotCodes so each barcode can have a unique
        # identifier
        allSpotCodes = []
        allBarcodes = []
        allNeighbors = list(data['neighbors'])[rang[0]: rang[1]]
        for i in range(len(allNeighbors)):
            neighbors = deepcopy(allNeighbors[i])
            neighborLists = []
            for rnd in range(roundNum):
                # Adds a -1 to each round of the neighbors dictionary (allows barcodes with dropped
                # rounds to be created)
                if roundOmitNum > 0:
                    neighbors[rnd].append(-1)
                neighborLists.append(neighbors[rnd])
            # Creates all possible spot code combinations from neighbors
            codes = list(product(*neighborLists))
            # Only save the ones with the correct number of dropped rounds
            spotCodes = [code for code in codes if Counter(code)[-1] == roundOmitNum]
            # Create barcodes from spot codes using the mapping from spot ID to channel
            barcodes = []
            for spotCode in spotCodes:
                barcode = []
                for spotInd in range(len(spotCode)):
                    if spotCode[spotInd] == -1:
                        barcode.append(-1)
                    else:
                        barcode.append(channelDict[spotInd][spotCode[spotInd]])
                barcodes.append(tuple(barcode))

            allBarcodes.append(barcodes)
            allSpotCodes.append(spotCodes)

        return (allSpotCodes, allBarcodes)

    # Only keep spots that have enough neighbors to form a barcode (determined by the total number
    # of rounds and the number of rounds that can be omitted from each code)
    passingSpots = {}
    roundNum = len(neighborDict)
    for key in neighborDict[currentRound]:
        if len(neighborDict[currentRound][key]) >= roundNum - roundOmitNum:
            passingSpots[key] = neighborDict[currentRound][key]
    passed = list(passingSpots.keys())
    roundData = roundData.iloc[passed]
    roundData['neighbors'] = [passingSpots[i] for i in roundData.index]
    roundData = roundData.reset_index(drop=True)

    # Find all possible barcodes for the spots in each round by splitting each round's spots into
    # numJob chunks and constructing each chunks barcodes in parallel

    # Save the current round's data table and the channelDict to ray memory
    dataID = ray.put(roundData)
    channelDictID = ray.put(channelDict)

    # Calculates index ranges to chunk data by
    ranges = [0]
    for i in range(1, numJobs + 1):
        ranges.append(int((len(roundData) / numJobs) * i))

    # Run in parallel
    results = [barcodeBuildFunc.remote(dataID, channelDictID, (ranges[i], ranges[i + 1]),
                                       roundOmitNum, roundNum)
               for i in range(len(ranges[:-1]))]
    rayResults = ray.get(results)

    # Add possible barcodes and spot codes (same order) to spot table (must chain results from
    # different jobs together)
    roundData['spot_codes'] = list(chain(*[job[0] for job in rayResults]))
    roundData['barcodes'] = list(chain(*[job[1] for job in rayResults]))

    return roundData

def decoder(roundData: pd.DataFrame,
            codebook: Codebook,
            roundOmitNum: int,
            currentRound: int,
            numJobs: int) -> pd.DataFrane:

    '''
    Function that takes spots tables with possible barcodes added and matches each to the codebook
    to identify any matches. Matches are added to the spot tables and spots without any matches are
    dropped

    Parameters
    ----------
        roundData : pd.DataFrane
            Modified spot table containing all possible barcodes that can be made from each spot
            for the current round

        codebook : Codebook
            starFISH Codebook object containg the barcode information for the experiment

        roundOmitNum : int
            Number of rounds that can be dropped from each barcode

        currentRound : int
            Current round being for which spots are being decoded

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        pd.DataFrane : Modified spot table with added columns with information on decodable
                       barcodes
    '''

    def generateRoundPermutations(size: int, roundOmitNum: int) -> list:
        '''
        Creates list of lists of logicals detailing the rounds to be used for decoding based on the
        current roundOmitNum

        Parameters
        ----------
            size : int
                Number of rounds in experiment

            roundOmitNum: int
                Number of rounds that can be dropped from each barcode

        Returns
        -------
            list : list of lists of logicals detailing the rounds to be used for decoding based on
                   the current roundOmitNum
        '''
        if roundOmitNum == 0:
            return [tuple([True] * size)]
        else:
            return sorted(set(list(permutations([*([False] * roundOmitNum),
                                                *([True] * (size - roundOmitNum))]))))

    @ray.remote
    def decodeFunc(data: pd.DataFrame,
                   roundPermutations: list,
                   permutationCodes: dict,
                   rnd: int) -> tuple:

        '''
        Subfunction for decoder that allows it to run in parallel chunks using ray

        Parameters
        ----------
            data : pd.DataFrame
                Spot table for the current round

            roundPermutations : list
                List of logicals from generateRoundPermutations that details the rounds to use in
                decoding

            permutationCodes : dict
                Dictionary containing barcode information for each roundPermutation

            rnd : int
                Current round being decoded

        Returns
        -------
            tuple : First element is a list of all decoded targets, second element is a list of all
                    decoded barcodes,third element is a list of all decoded spot codes, and the
                    fourth element is a list of rounds that were omitted for each decoded barcode
        '''

        # Goes through all possible decodings of each spot (ensures each spot is only looked up
        # once)
        allTargets = []
        allDecodedBarcodes = []
        allDecodedSpotCodes = []
        allRoundOmit = []
        allBarcodes = list(data['barcodes'])
        allSpotCodes = list(data['spot_codes'])
        for i in range(len(allBarcodes)):
            targets = []
            decodedBarcodes = []
            decodedSpotCodes = []
            roundOmit = []
            fullBarcodes = allBarcodes[i]
            fullSpotCodes = allSpotCodes[i]

            for currentRounds in roundPermutations:

                # Set omittedRound to the round being dropped, if no round is dropped omittedRound
                # becomes -1
                if 0 in currentRounds:
                    omittedRound = np.argwhere([not cr for cr in currentRounds])[0][0]
                else:
                    omittedRound = -1

                # Only try to decode barcodes for this spot if the current round is not the omitted
                # round
                if rnd != omittedRound:
                    # Modify spot codes and barcodes so that they match the current set of rounds
                    # being used for decoding
                    if omittedRound != -1:
                        spotCodes = [code for code in
                                     np.asarray([np.asarray(spotCode)[list(currentRounds)]
                                                 for spotCode in fullSpotCodes]) if -1 not in code]
                        barcodes = [code for code in
                                    np.asarray([np.asarray(barcode)[list(currentRounds)]
                                                for barcode in fullBarcodes]) if -1 not in code]
                    else:
                        spotCodes = fullSpotCodes
                        barcodes = fullBarcodes
                    # If all barcodes omit a round other than omittedRound, barcodes will be empty
                    if len(barcodes) > 0:
                        # Tries to find a match to each possible barcode from the spot
                        for j, barcode in enumerate(barcodes):
                            try:
                                # Try to assign target by using barcode as key in permutationsCodes
                                # dictionary for current set of rounds. If there is no barcode
                                # match, it will error and go to the except and if it succeeds it
                                # will add the data to the other lists for this barcode
                                targets.append(permutationCodes[currentRounds][tuple(barcode)])
                                decodedBarcodes.append(barcode)
                                decodedSpotCodes.append(list(spotCodes[j]))
                                roundOmit.append(omittedRound)
                            except Exception:
                                pass
            allTargets.append(targets)
            allDecodedBarcodes.append(decodedBarcodes)
            allDecodedSpotCodes.append(decodedSpotCodes)
            allRoundOmit.append(roundOmit)

        return (allTargets, allDecodedBarcodes, allDecodedSpotCodes, allRoundOmit)

    # Create list of logical arrays corresponding to the round sets being used to decode
    roundPermutations = generateRoundPermutations(codebook.sizes[Axes.ROUND], roundOmitNum)

    # Create dictionary where the keys are the different round sets that can be used for decoding
    # and the values are the modified codebooks corresponding to the rounds used
    permCodeDict = {}
    for currentRounds in roundPermutations:
        codes = codebook.argmax(Axes.CH.value)
        currentCodes = codes.sel(r=list(currentRounds))
        currentCodes.values = np.ascontiguousarray(currentCodes.values)
        permCodeDict[currentRounds] = dict(zip([tuple(code) for code in currentCodes.data],
                                               currentCodes['target'].data))

    # Goes through each round in filtered_prsr and tries to decode each spot's barcodes

    # Put data table and permutations codes dictionary in ray storage
    permutationCodesID = ray.put(permCodeDict)

    # Calculates index ranges to chunk data by and creates list of chunked data to loop through
    ranges = [0]
    for i in range(1, numJobs + 1):
        ranges.append(int((len(roundData) / numJobs) * i))
    chunkedData = []
    for i in range(len(ranges[:-1])):
        chunkedData.append(deepcopy(roundData[ranges[i]:ranges[i + 1]]))

    # Run in parallel
    results = [decodeFunc.remote(chunkedData[i], roundPermutations, permutationCodesID,
                                 currentRound) for i in range(len(ranges[:-1]))]
    rayResults = ray.get(results)

    # Update table
    roundData['targets'] = list(chain(*[job[0] for job in rayResults]))
    roundData['decoded_barcodes'] = list(chain(*[job[1] for job in rayResults]))
    roundData['decoded_spot_codes'] = list(chain(*[job[2] for job in rayResults]))
    roundData['omitted_round'] = list(chain(*[job[3] for job in rayResults]))

    # Drop barcodes and spot_codes column (saves memory)
    roundData = roundData.drop(['neighbors', 'spot_codes', 'barcodes'], axis=1)

    # Remove rows that have no decoded barcodes
    roundData = roundData[roundData['targets'].astype(bool)].reset_index(drop=True)

    # Add -1 spacer back into partial barcodes/spot codes so we can easily tell which round each
    # spot ID is from
    if roundOmitNum > 0:
        allBarcodes = []
        allSpotCodes = []
        dataBarcodes = roundData['decoded_barcodes']
        dataSpotCodes = roundData['decoded_spot_codes']
        dataOmittedRounds = roundData['omitted_round']
        for i in range(len(roundData)):
            barcodes = [list(code) for code in dataBarcodes[i]]
            spotCodes = [list(code) for code in dataSpotCodes[i]]
            omittedRounds = dataOmittedRounds[i]
            barcodes = [barcodes[j][:omittedRounds[j]] + [-1] + barcodes[j][omittedRounds[j]:]
                        for j in range(len(barcodes))]
            spotCodes = [spotCodes[j][:omittedRounds[j]] + [-1] + spotCodes[j][omittedRounds[j]:]
                         for j in range(len(barcodes))]
            allBarcodes.append(barcodes)
            allSpotCodes.append(spotCodes)
        roundData['decoded_barcodes'] = allBarcodes
        roundData['decoded_spot_codes'] = allSpotCodes

    return roundData

def distanceFilter(roundData: pd.DataFrame,
                   spotCoords: dict,
                   currentRound: int,
                   numJobs: int) -> pd.DataFrame:
    '''
    Function that chooses between the best barcode for each spot from the set of decodable barcodes.
    Does this by choosing the barcode with the least spatial variance among the spots that make it
    up. If there is a tie, the spot is dropped as ambiguous.

    Parameters
    ----------
        roundData : pd.DataFrame
            Modified spot table containing info on decodable barcodes for the spots in the current
            round

        spotCoords : dict
            Dictionary containing spatial coordinates of spots in each round indexed by their IDs

        currentRound : int
            Current round number to calculate distances for

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        pd.DataFrame : Modified spot table with added columns to with info on the "best" barcode
                       found for each spot
    '''

    @ray.remote
    def distanceFunc(subSpotCodes: list, spotCoords: dict) -> list:
        '''
        Subfunction for distanceFilter to allow it to run in parallel using ray

        Parameters
        ----------
            subSpotCodes : list
                Chunk of full list of spot codes for the current round to calculate the spatial
                variance for

            spotCoords : dict
                Dictionary containing spatial locations for spots by their IDs in the original
                spotTables object

        Returns
        -------
            list: list of spatial variances for the current chunk of spot codes

        '''

        # Calculate spatial variances for current chunk of spot codes
        allDistances = []
        for spotCodes in subSpotCodes:
            distances = []
            for s, spotCode in enumerate(spotCodes):
                coords = []
                for j, spot in enumerate(spotCode):
                    if spot != -1:
                        # Extract spot coordinates from spotCoords
                        z = spotCoords[j][spot]['z']
                        y = spotCoords[j][spot]['y']
                        x = spotCoords[j][spot]['x']
                        coords.append([z, y, x])
                coords = np.asarray(coords)
                # Distance is calculate as the sum of variances of the coordinates along each axis
                distances.append(sum(np.var(coords, axis=0)))
            allDistances.append(distances)
        return allDistances

    # Calculate the spatial variance for each decodable barcode for each spot in each round
    allSpotCodes = roundData['decoded_spot_codes']

    # Put spotCoords dictionary into ray memory
    spotCoordsID = ray.put(spotCoords)

    # Calculates index ranges to chunk data by
    ranges = [0]
    for i in range(1, numJobs):
        ranges.append(int((len(roundData) / numJobs) * i))
    ranges.append(len(roundData))
    chunkedSpotCodes = [allSpotCodes[ranges[i]:ranges[i + 1]] for i in range(len(ranges[:-1]))]

    # Run in parallel using ray
    results = [distanceFunc.remote(subSpotCodes, spotCoordsID) for subSpotCodes
               in chunkedSpotCodes]
    rayResults = ray.get(results)

    # Add distances to decodedTables as new column
    roundData['distance'] = list(chain(*[job for job in rayResults]))

    # Pick minimum distance barcode(s) for each spot
    bestSpotCodes = []
    bestBarcodes = []
    bestTargets = []
    bestDistances = []
    dataSpotCodes = list(roundData['decoded_spot_codes'])
    dataBarcodes = list(roundData['decoded_barcodes'])
    dataDistances = list(roundData['distance'])
    dataTargets = list(roundData['targets'])
    for i in range(len(roundData)):
        spotCodes = dataSpotCodes[i]
        barcodes = dataBarcodes[i]
        distances = dataDistances[i]
        targets = dataTargets[i]
        # If only one barcode to choose from, that one is picked as best
        if len(distances) == 1:
            bestSpotCodes.append(spotCodes)
            bestBarcodes.append(barcodes)
            bestTargets.append(targets)
            bestDistances.append(distances)
        # Otherwise find the minimum, and if there are multiple minimums
        else:
            minDist = 100
            minCount = 0
            for d, distance in enumerate(distances):
                if distance < minDist:
                    minDist = distance
                    minCount = 1
                    minInds = []
                    minInds.append(d)
                elif distance == minDist:
                    minCount += 1
                    minInds.append(d)
            bestSpotCodes.append([spotCodes[i] for i in range(len(spotCodes)) if i in minInds])
            bestBarcodes.append([barcodes[i] for i in range(len(barcodes)) if i in minInds])
            bestTargets.append([targets[i] for i in range(len(targets)) if i in minInds])
            bestDistances.append([distances[i] for i in range(len(distances)) if i in minInds])
    # Create new columns with minimum distance barcode information
    roundData['best_spot_codes'] = bestSpotCodes
    roundData['best_barcodes'] = bestBarcodes
    roundData['best_targets'] = bestTargets
    roundData['best_distances'] = bestDistances

    # Drop old columns
    roundData = roundData.drop(['targets', 'decoded_barcodes', 'decoded_spot_codes',
                                'omitted_round'], axis=1)

    # Only keep barcodes with only one minimum distance
    keep = []
    barcodes = roundData['best_barcodes']
    for i in range(len(roundData)):
        if len(barcodes[i]) == 1:
            keep.append(i)
    roundData = roundData.iloc[keep]

    return roundData

def cleanup(bestPerSpotTables: dict,
            spotCoords: dict,
            filterRounds: int) -> pd.DataFrame:

    '''
    Function that combines all "best" codes for each spot in each round into a single table,
    filters them by their frequency (with a user-defined threshold), chooses between overlapping
    codes (using the same distance function as used earlier), and finally adds some additional
    information to the final set of barcodes

    Parameters
    ----------
        bestPerSpotTables : dict
            Spot tables dictionary containing columns with information on the "best" barcode found
            for each spot

        spotCoords : dict
            Dictionary containing spatial locations of spots

        filterRounds : int
            Number of rounds that a barcode must be identified in to pass filters (higher = more
            stringent filtering), default = 1 - #rounds  or 1 - roundOmitNum if roundOmitNum > 0

    Returns
    -------
        pd.DataFrame : Dataframe containing final set of codes that have passed all filters

    '''

    # Create merged spot results dataframe containing the passing barcodes found in all the rounds
    mergedCodes = pd.DataFrame()
    roundNum = len(bestPerSpotTables)
    for r in range(roundNum):
        barcodes = bestPerSpotTables[r]['best_barcodes']
        spotCodes = bestPerSpotTables[r]['best_spot_codes']
        targets = bestPerSpotTables[r]['best_targets']
        distances = bestPerSpotTables[r]['best_distances']
        # Turn each barcode and spot code into a tuple so they can be used as dictionary keys
        bestPerSpotTables[r]['best_barcodes'] = [tuple(barcode[0]) for barcode in barcodes]
        bestPerSpotTables[r]['best_spot_codes'] = [tuple(spotCode[0]) for spotCode in spotCodes]
        bestPerSpotTables[r]['best_targets'] = [target[0] for target in targets]
        bestPerSpotTables[r]['best_distances'] = [distance[0] for distance in distances]
        mergedCodes = mergedCodes.append(bestPerSpotTables[r])
    mergedCodes = mergedCodes.reset_index(drop=True)

    # Only use codes that were found in >= filterRounds rounds
    spotCodes = mergedCodes['best_spot_codes']
    counts = defaultdict(int)  # type: dict
    for code in spotCodes:
        counts[code] += 1
    passing = list(set(code for code in counts if counts[code] >= filterRounds))
    finalCodes = mergedCodes[mergedCodes['best_spot_codes'].isin(passing)].reset_index(drop=True)
    finalCodes = finalCodes.iloc[finalCodes['best_spot_codes'].drop_duplicates().index]
    finalCodes = finalCodes.reset_index(drop=True)

    # Choose between overlapping spot codes based on which has the smaller spatial variance
    for r in range(roundNum):
        roundSpots = [code[r] for code in finalCodes['best_spot_codes'] if code[r] != -1]
        dupSpots = set([spot for spot in roundSpots if Counter(roundSpots)[spot] > 1])
        drop = []
        for spot in dupSpots:
            locs = np.where(np.asarray(roundSpots) == spot)[0]
            distances = [finalCodes.loc[loc, 'best_distances'] for loc in locs]
            minInd = np.where(distances == min(distances))[0]
            if len(minInd) > 1:
                drop.extend([ind for ind in minInd])
            else:
                drop.extend([locs[i] for i in range(len(locs)) if i != minInd])
        finalCodes = finalCodes.iloc[[i for i in range(len(finalCodes)) if i not in drop]]
        finalCodes = finalCodes.reset_index(drop=True)

    # Add spot coordinates, barcode center coordinates, and number of rounds used for each barcode
    # to table
    allCoords = []
    centers = []
    roundsUsed = []
    for i in range(len(finalCodes)):
        coords = []
        spotCode = finalCodes.iloc[i]['best_spot_codes']
        roundsUsed.append(roundNum - Counter(spotCode)[-1])
        for r in range(roundNum):
            if spotCode[r] != -1:
                z = spotCoords[r][spotCode[r]]['z']
                y = spotCoords[r][spotCode[r]]['y']
                x = spotCoords[r][spotCode[r]]['x']
                coords.append((x, y, z))
        allCoords.append(coords)
        coords = np.asarray([coord for coord in coords])
        center = np.asarray(coords).mean(axis=0)
        centers.append(center)
    finalCodes['coords'] = allCoords
    finalCodes['center'] = centers
    finalCodes['rounds_used'] = roundsUsed

    return finalCodes

def removeUsedSpots(finalCodes: pd.DataFrame, neighborDict: dict) -> dict:
    '''
    Remove spots found to be in barcodes for the current round omission number so they are not used
    for the next

    Parameters
    ----------
        finalCodes : pd.DataFrame
            Dataframe containing final set of codes that have passed all filters

        neighborDict : dict
            Dictionary that contains all the neighbors for each spot in other rounds that are
            within the search radius

    Returns
    -------
        dict : Modified version of neighborDict with spots that have been used in the current round
               omission removed
    '''

    # Remove used spots
    roundNum = len(neighborDict)
    for r in range(roundNum):
        usedSpots = list(set([passed[r] for passed in finalCodes['best_spot_codes']
                              if passed[r] != -1]))
        for spot in usedSpots:
            for key in neighborDict[r][spot]:
                for neighbor in neighborDict[r][spot][key]:
                    neighborDict[key][neighbor][r] = [i for i in neighborDict[key][neighbor][r]
                                                      if i != spot]
            del neighborDict[r][spot]

    # Remove empty lists
    for r in range(roundNum):
        for spot in neighborDict[r]:
            for key in [*neighborDict[r][spot]]:
                if neighborDict[r][spot][key] == []:
                    del neighborDict[r][spot][key]

    return neighborDict
