import typing
import warnings
from collections import Counter, defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from itertools import chain, islice, permutations, product

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from starfish.core.codebook.codebook import Codebook
from starfish.types import Axes

warnings.filterwarnings('ignore')

def findNeighbors(spotTables: dict,
                  searchRadius: float,
                  numJobs: int) -> dict:

    '''
    Using scipy's cKDTree method, finds all neighbors within the seach radius between the spots in
    each pair of rounds and stores the indices in a dictionary for later access.

    Parameters
    ----------
        spotTables : dict
            Dictionary with round labels as keys and pandas dataframes containing spot information
            for its key round as values (result of _merge_spots_by_round function)

        searchRadius : float
            Distance that spots can be from each other and still form a barcode

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        dict: a dictionary with the following structure:
            {(round1, round2): index table showing neighbors between spots in round1 and round2
             where round1 != round2}
    '''

    allNeighborDict = {}
    for r1 in range((len(spotTables))):
        tree = cKDTree(spotTables[r1][['z', 'y', 'x']])
        for r2 in list(range((len(spotTables))))[r1 + 1:]:
            allNeighborDict[(r1, r2)] = tree.query_ball_point(spotTables[r2][['z', 'y', 'x']],
                                                              searchRadius, workers=numJobs)

    return allNeighborDict

def createNeighborDict(spotTables: dict,
                       searchRadius: float,
                       neighborsByRadius: dict) -> dict:

    '''
    Create dictionary of neighbors (within the search radius) in other rounds for each spot.

    Parameters
    ----------

        spotTables : dict
            Dictionary with round labels as keys and pandas dataframes containing spot information
            for its key round as values (result of _merge_spots_by_round function)

        searchRadius : float
            Distance that spots can be from each other and still form a barcode

        neighborsByRadius : dict
            Dictionary of outputs from findNeighbors() where each key is a radius and the value is
            the findNeighbors dictionary

    Returns
    -------

        dict: a dictionary with the following structure
            neighborDict[roundNum][spotID] = {0 : neighbors in round 0, 1: neighbors in round 1,etc}
    '''

    # Create empty neighbor dictionary
    neighborDict = {}
    spotIDs = {}
    for r in spotTables:
        spotIDs[r] = {idd: 0 for idd in spotTables[r]['spot_id']}
        neighborDict[r] = {i: defaultdict(list, {r: [i]}) for i in spotTables[r]['spot_id']}

    # Add neighbors in neighborsByRadius[searchRadius] but check to make sure that spot is still
    # available before adding it
    for r1 in range(len(spotTables)):
        for r2 in list(range((len(spotTables))))[r1 + 1:]:
            for j, neighbors in enumerate(neighborsByRadius[searchRadius][(r1, r2)]):
                try:
                    spotIDs[r2][j + 1]
                    for neighbor in neighbors:
                        try:
                            spotIDs[r1][neighbor + 1]
                            neighborDict[r1][neighbor + 1][r2].append(j + 1)
                            neighborDict[r2][j + 1][r1].append(neighbor + 1)
                        except Exception:
                            pass
                except Exception:
                    pass
    return neighborDict

def createRefDicts(spotTables: dict, numJobs: int) -> tuple:

    '''
    Create dictionaries with mapping from spot id (row index + 1) in spotTables to channel label,
    spatial coordinates raw intensity and normalized intensity.

    Parameters
    ----------
        spotTables : dict
            Dictionary with round labels as keys and pandas dataframes containing spot information
            for its key round as values (result of _merge_spots_by_round function)

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        tuple : First object is the channel dictionary, second is the spatial coordinate dictionary,
                the third object is the raw spot instensity dictionary, and the last object is the
                normalized spot intensity dictionary
    '''

    # Create channel label and spatial coordinate dictionaries
    channelDict = {}
    spotCoords = {}
    for r in [*spotTables]:
        channelDict[r] = spotTables[r][['c', 'spot_id']].set_index('spot_id').to_dict()['c']
        channelDict[r][0] = 0
        tmpTable = spotTables[r][['z', 'y', 'x', 'spot_id']].set_index('spot_id')
        spotCoords[r] = tmpTable.to_dict(orient='index')
        for key in [*spotCoords[r]]:
            spotCoords[r][key] = tuple(spotCoords[r][key].values())

    # Create raw intensity dictionary
    spotIntensities = {r: spotTables[r][['intensity', 'spot_id']].set_index('spot_id').to_dict()
                       ['intensity'] for r in [*spotTables]}
    for r in [*spotTables]:
        spotIntensities[r][0] = 0

    # Create normalized intensity dictionary
    spotQualDict = spotQuality(spotTables, spotCoords, spotIntensities, channelDict, numJobs)

    return channelDict, spotCoords, spotIntensities, spotQualDict

def encodeSpots(spotCodes: list) -> list:

    '''
    For compressing spot ID codes into single integers. Saves memory. The number of digits in
    each ID is counted and these integer lengths and concatenated into a string in the same
    order as the IDs they correspond to. The IDs themselves are then converted to strings and
    concatenated to this, also maintaining order.

    Parameters
    ----------
        spotCodes : list
            List of spot codes (each a tuple of integers with length equal to the number of rounds)


    Returns
    -------
        list: List of compressed spot codes, one int per code
    '''

    strs = [list(map(str, code)) for code in spotCodes]
    compressed = [int(''.join(map(str, map(len, intStr))) + ''.join(intStr)) for intStr in strs]

    return compressed

def decodeSpots(compressed: list, roundNum: int) -> list:

    '''
    Reconverts compressed spot codes back into their roundNum length tupes of integers with
    the same order and IDs as their original source. First roundNum values in the compressed
    code will each correspond to the string length of each spot ID integer (as long as no round
    has 10 billion or more spots). Can use these to determine how to split the rest of the string
    to retrieve the original values in the correct order.

    Parameters
    ----------
        compressed : list
            List of integer values corresponding to compressed spot codes

        roundNum : int
            The number of rounds in the experiment

    Returns
    -------
        list: List of recovered spot codes in their original tuple form
    '''

    strs = [str(intStr) for intStr in compressed]
    idxs, nums = list(zip(*[(map(int, s[:roundNum]), [iter(s[roundNum:])] * roundNum)
                            for s in strs]))
    decompressed = [tuple(int(''.join(islice(n, i))) for i, n in zip(idxs[j], nums[j]))
                    for j in range(len(idxs))]
    return decompressed

def spotQualityFunc(spots: list,
                    spotCoords: dict,
                    spotIntensities: dict,
                    spotTables: dict,
                    channelDict: dict,
                    r: int) -> list:

    '''
    Helper function for spotQuality to run in parallel

    Parameters
    ----------
        spots : list
            List of spot IDs in the current round to calculate the normalized intensity of

        spotCoords : dict
            Spot ID to spatial coordinate dictionary

        spotIntensities : dict
            Spot ID to raw intensity dictionary

        spotTables : dict
            Dictionary containing spot info tables

        channelDict : dict
            Spot ID to channel label dictionary

        r : int
            Current round

    Returns
    -------
        list : list of normalized spot intensities of the input spot IDs
    '''

    # Find spots in the same neighborhood (same channel and z slice and less than 100 pixels away
    # in either x or y direction)
    neighborhood = 100
    quals = []
    for i, spot in enumerate(spots):
        z, y, x = spotCoords[r][spot]
        ch = channelDict[r][spot]
        yMin = y - neighborhood if y - neighborhood >= 0 else 0
        yMax = y + neighborhood if y + neighborhood <= 2048 else 2048
        xMin = x - neighborhood if x - neighborhood >= 0 else 0
        xMax = x + neighborhood if x + neighborhood <= 2048 else 2048
        neighborInts = spotTables[r][(spotTables[r]['c'] == ch)
                                     & (spotTables[r]['z'] == z)
                                     & (spotTables[r]['y'] >= yMin)
                                     & (spotTables[r]['y'] < yMax)
                                     & (spotTables[r]['x'] >= xMin)
                                     & (spotTables[r]['x'] < xMax)]['intensity']
        # If no neighbors drop requirement that they be within 100 pixels of each other
        if len(neighborInts) == 1:
            neighborInts = spotTables[r][(spotTables[r]['c'] == ch)
                                         & (spotTables[r]['z'] == z)]['intensity']
        # If still no neighbors drop requirement that they be on the same z slice
        if len(neighborInts) == 1:
            neighborInts = spotTables[r][(spotTables[r]['c'] == ch)]['intensity']
        # Calculate the l2 norm of the neighbor's intensities and divide the spot's intensity by
        # this value to get it's normalized intensity value
        norm = np.linalg.norm(neighborInts)
        quals.append(spotIntensities[r][spot] / norm)

    return quals

def spotQuality(spotTables: dict,
                spotCoords: dict,
                spotIntensities: dict,
                channelDict: dict,
                numJobs: int) -> dict:

    '''
    Creates dictionary mapping each spot ID to their normalized intensity value. Calculated as the
    spot intensity value divided by the l2 norm of the intensities of all the spots in the same
    neighborhood.

    Parameters
    ----------
        spotTables : dict
            Dictionary containing spot info tables

        spotCoords : dict
            Spot ID to spatial coordinate dictionary

        spotIntensities : dict
            Spot ID to raw intensity dictionary

        channelDict : dict
            Spot ID to channel label dictionary

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        dict : dictionary mapping spot ID to it's normalized intensity value
    '''

    # Calculate normalize spot intensities for each spot in each round
    spotQuals = {}  # type: dict
    for r in range(len(spotTables)):
        roundSpots = spotTables[r]['spot_id']
        spotQuals[r] = {}

        # Calculates index ranges to chunk data by
        ranges = [0]
        for i in range(1, numJobs):
            ranges.append(int((len(roundSpots) / numJobs) * i))
        ranges.append(len(roundSpots))
        chunkedSpots = [roundSpots[ranges[i]:ranges[i + 1]] for i in range(len(ranges[:-1]))]

        # Run in parallel
        with ProcessPoolExecutor() as pool:
            part = partial(spotQualityFunc, spotCoords=spotCoords, spotIntensities=spotIntensities,
                           spotTables=spotTables, channelDict=channelDict, r=r)
            poolMap = pool.map(part, [subSpots for subSpots in chunkedSpots])
            results = [x for x in poolMap]

        # Extract results
        for spot, qual in zip(roundSpots, list(chain(*results))):
            spotQuals[r][spot] = qual

    return spotQuals

def barcodeBuildFunc(allNeighbors: list,
                     channelDict: dict,
                     currentRound: int,
                     roundOmitNum: int,
                     roundNum: int) -> list:

    '''
    Subfunction to buildBarcodes that allows it to run in parallel chunks

    Parameters
    ----------
        allNeighbors : list
            List of neighbor from which to build barcodes from

        channelDict : dict
            Dictionary mapping spot IDs to their channels labels

        currentRound : int
            The round that the spots being used for reference points are found in

        roundOmitNum : int
            Maximum hamming distance a barcode can be from it's target in the codebook and
            still be uniquely identified (i.e. number of error correction rounds in each
            the experiment)

        roundNum : int
            Total number of round in experiment

    Returns
    -------
        list : list of the possible spot codes
    '''

    # spotCodes are the ordered spot IDs of the spots making up each barcode while barcodes are
    # the corresponding channel labels, need spotCodes so each barcode can have a unique
    # identifier
    allSpotCodes = []
    for neighbors in allNeighbors:
        neighborLists = [neighbors[rnd] for rnd in range(roundNum)]
        # Adds a 0 to each round of the neighbors dictionary (allows barcodes with dropped
        # rounds to be created)
        if roundOmitNum > 0:
            [neighbors[rnd].append(0) for rnd in range(roundNum)]
        # Creates all possible spot code combinations from neighbors
        codes = list(product(*neighborLists))
        # Only save the ones with the correct number of dropped rounds
        counters = [Counter(code) for code in codes]  # type: typing.List[Counter]
        spotCodes = [code for j, code in enumerate(codes) if counters[j][0] == roundOmitNum]
        spotCodes = [code for code in spotCodes if code[currentRound] != 0]

        allSpotCodes.append(encodeSpots(spotCodes))

    return allSpotCodes

def buildBarcodes(roundData: pd.DataFrame,
                  neighborDict: dict,
                  roundOmitNum: int,
                  channelDict: dict,
                  strictness: int,
                  currentRound: int,
                  numJobs: int) -> pd.DataFrame:

    '''
    Builds possible barcodes for each seed spot from its neighbors. First checks that each spot has
    enough neighbors in each round to form a barcode and, depending on the strictness value, drops
    spots who have too many possible barcodes to choose from

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

        strictness: int
            Determines the number of possible codes a spot is allowed to have before it is dropped
            as ambiguous (if it is positive)

        currentRound : int
            Current round to build barcodes for (same round that roundData is from)

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        pd.DataFrame : Copy of roundData with an additional column which lists all the possible spot
                       codes that could be made from each spot's neighbors for those spots that
                       passed the strictness requirement (if it is positive)
    '''

    # Only keep spots that have enough neighbors to form a barcode (determined by the total number
    # of rounds and the number of rounds that can be omitted from each code) and if strictness is
    # positive, drop spots that have more than the strictness value number of possible barcodes
    roundNum = len(neighborDict)
    if strictness > 0:
        passed = [key for key in neighborDict[currentRound] if
                  len(neighborDict[currentRound][key]) >= roundNum - roundOmitNum
                  and np.prod([len(values) for values in
                               neighborDict[currentRound][key].values()]) <= strictness]
    else:
        passed = [key for key in neighborDict[currentRound] if
                  len(neighborDict[currentRound][key]) >= roundNum - roundOmitNum]
    roundData = roundData[roundData['spot_id'].isin(passed)].reset_index(drop=True)
    roundData['neighbors'] = [neighborDict[currentRound][p] for p in passed]

    # Find all possible barcodes for the spots in each round by splitting each round's spots into
    # numJob chunks and constructing each chunks barcodes in parallel

    # Calculates index ranges to chunk data by
    ranges = [0]
    for i in range(1, numJobs + 1):
        ranges.append(int((len(roundData) / numJobs) * i))
    chunkedNeighbors = [list(roundData['neighbors'])[ranges[i]: ranges[i + 1]] for i in
                        range(len(ranges[:-1]))]

    # Run in parallel
    with ProcessPoolExecutor() as pool:
        part = partial(barcodeBuildFunc, channelDict=channelDict, currentRound=currentRound,
                       roundOmitNum=roundOmitNum, roundNum=roundNum)
        poolMap = pool.map(part, [chunkedNeighbors[i] for i in range(len(chunkedNeighbors))])
        results = [x for x in poolMap]

    # Drop unneeded columns (saves memory)
    roundData = roundData.drop(['neighbors', 'spot_id'], axis=1)

    # Add possible spot codes to spot table (must chain results from different jobs together)
    roundData['spot_codes'] = list(chain(*[job for job in results]))

    return roundData

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

def decodeFunc(data: pd.DataFrame, permutationCodes: dict) -> tuple:

    '''
    Subfunction for decoder that allows it to run in parallel chunks

    Parameters
    ----------
        data : pd.DataFrame
            DataFrame with columns called 'barcodes' and 'spot_codes'

        permutationCodes : dict
            Dictionary containing barcode information for each roundPermutation

    Returns
    -------
        tuple : First element is a list of all decoded targets, second element is a list of all
                decoded spot codes
    '''

    # Checks if each barcode is in the permutationsCodes dict, if it isn't, there is no match
    allTargets = []
    allDecodedSpotCodes = []
    allBarcodes = list(data['barcodes'])
    allSpotCodes = list(data['spot_codes'])
    for i in range(len(allBarcodes)):
        targets = []
        decodedSpotCodes = []
        for j, barcode in enumerate(allBarcodes[i]):
            try:
                # Try to assign target by using barcode as key in permutationsCodes dictionary for
                # current set of rounds. If there is no barcode match, it will error and go to the
                # except and if it succeeds it will add the data to the other lists for this barcode
                targets.append(permutationCodes[barcode])
                decodedSpotCodes.append(allSpotCodes[i][j])
            except Exception:
                pass
        allTargets.append(targets)
        allDecodedSpotCodes.append(decodedSpotCodes)
    return (allTargets, allDecodedSpotCodes)

def decoder(roundData: pd.DataFrame,
            codebook: Codebook,
            channelDict: dict,
            strictness: int,
            currentRoundOmitNum: int,
            currentRound: int,
            numJobs: int) -> pd.DataFrame:

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

        channelDict : dict
            Dictionary with mappings between spot IDs and their channel labels

        strictness : int
            Determines the number of target matching barcodes each spot is allowed before it is
            dropped as ambiguous (if it is negative)

        currentRoundOmitNum : int
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

    # Add barcodes column by mapping spotIDs in spot_codes to channel labels using channelDict
    if strictness > 0:
        roundData['barcodes'] = [[hash(tuple([channelDict[j][spot] for j, spot in
                                  enumerate(code)]))] for code in roundData['spot_codes']]
        roundData['spot_codes'] = [[codes] for codes in roundData['spot_codes']]
    else:
        barcodes = []
        for codes in roundData['spot_codes']:
            barcodes.append([hash(tuple([channelDict[j][spot] for j, spot in enumerate(code)]))
                             for code in decodeSpots(codes, len(channelDict))])
        roundData['barcodes'] = barcodes

    # Create list of logical arrays corresponding to the round sets being used to decode
    roundPermutations = generateRoundPermutations(codebook.sizes[Axes.ROUND], currentRoundOmitNum)

    # Create dictionary where the keys are the different round sets that can be used for decoding
    # and the values are the modified codebooks corresponding to the rounds used
    permCodeDict = {}
    for currentRounds in roundPermutations:
        codes = codebook.argmax(Axes.CH.value)
        if currentRoundOmitNum > 0:
            omittedRounds = np.argwhere(~np.asarray(currentRounds))
            codes.data[:, omittedRounds] = -1
        codes.data += 1
        roundDict = dict(zip([hash(tuple(code)) for code in codes.data], codes['target'].data))
        permCodeDict.update(roundDict)

    # Calculates index ranges to chunk data by and creates list of chunked data to loop through
    ranges = [0]
    for i in range(1, numJobs + 1):
        ranges.append(int((len(roundData) / numJobs) * i))
    chunkedData = []
    for i in range(len(ranges[:-1])):
        chunkedData.append(deepcopy(roundData[ranges[i]:ranges[i + 1]]))

    # Run in parallel
    with ProcessPoolExecutor() as pool:
        part = partial(decodeFunc, permutationCodes=permCodeDict)
        poolMap = pool.map(part, [chunkedData[i] for i in range(len(chunkedData))])
        results = [x for x in poolMap]

    # Update table
    roundData['targets'] = list(chain(*[job[0] for job in results]))
    roundData['spot_codes'] = list(chain(*[job[1] for job in results]))

    roundData = roundData[[len(targets) > 0 for targets in
                           roundData['targets']]].reset_index(drop=True)

    if len(roundData) > 0:
        if strictness < 0:
            roundData = roundData[[len(targets) <= np.abs(strictness) for targets in
                                   roundData['targets']]].reset_index(drop=True)

            roundData = roundData.drop(['barcodes'], axis=1)

    return roundData

def distanceFunc(spotsAndTargets: list,
                 spotCoords: dict,
                 spotQualDict: dict,
                 currentRoundOmitNum: int) -> tuple:

    '''
    Subfunction for distanceFilter to allow it to run in parallel

    Parameters
    ----------
        subSpotCodes : list
            Chunk of full list of spot codes for the current round to calculate the spatial
            variance for

        subSpotCodes : list
            Chunk of full list of targets (0s if strictness is positive) associated with the
            current set of spots whose spatial variance is being calculated

        spotCoords : dict
            Spot ID to spatial coordinate dictionary

        spotQualDict : dict
            Spot ID to normalized intensity value dictionary

        currentRoundOmitNum : int
            Number of rounds that can be dropped from each barcode

    Returns
    -------
        tuple: First object is the min scoring spot code for each spots, the second is the min
               score for each spot, and the third is the min scoring target for each spot
    '''

    subSpotCodes = spotsAndTargets[0]
    subTargets = spotsAndTargets[1]

    # Find minimum scoring combination of spots from set of possible combinations
    constant = 2
    bestSpotCodes = []
    bestDistances = []
    bestTargets = []
    for i, codes in enumerate(subSpotCodes):
        quals = [sum([spotQualDict[r][spot] for r, spot in enumerate(code) if spot != 0])
                 for code in codes]
        newQuals = np.asarray([-np.log(1 / (1 + (len(spotCoords) - currentRoundOmitNum - qual)))
                               for qual in quals])
        subCoords = [[spotCoords[r][spot] for r, spot in enumerate(code) if spot != 0]
                     for code in codes]
        spaVars = [sum(np.var(np.asarray(coords), axis=0)) for coords in subCoords]
        newSpaVars = np.asarray([-np.log(1 / (1 + spaVar)) for spaVar in spaVars])
        combined = newQuals + (newSpaVars * constant)
        minInds = np.where(combined == min(combined))[0]
        if len(minInds) == 1:
            bestSpotCodes.append(codes[minInds[0]])
            bestDistances.append(combined[minInds[0]])
            bestTargets.append(subTargets[i][minInds[0]])
        else:
            bestSpotCodes.append(-1)
            bestDistances.append(-1)
            bestTargets.append(-1)

    return (bestSpotCodes, bestDistances, bestTargets)

def distanceFilter(roundData: pd.DataFrame,
                   spotCoords: dict,
                   spotQualDict: dict,
                   currentRound: int,
                   currentRoundOmitNum: int,
                   numJobs: int) -> pd.DataFrame:

    '''
    Function that chooses between the best barcode for each spot from the set of decodable barcodes.
    Does this by choosing the barcode with the least spatial variance and high intensity spots
    according to this calculation:

    Score = -log(1 / 1 + (numRounds - qualSum)) + (-log(1 / 1 + spaVar) * constant)
    Where:
        numRounds = number of rounds being used for decoding (total - currentRoundOmitNum)
        qualSum = sum of normalized intensity values for the spots in the code
        spaVar = spatial variance of spots in code, calculates as the sum of variances of the
                 values in each spatial dimension
        constant = a constant that determines the balance between the score being more influenced
                   by spatial variance or intensity, set to 2 so spatial variance is the biggest
                   deciding factor but allows ties to be broken by intensity

    Parameters
    ----------
        roundData : pd.DataFrame
            Modified spot table containing info on decodable barcodes for the spots in the current
            round

        spotCoords : dict
            Spot ID to spatial coordinate dictionary

        spotQualDict : dict
            Spot ID to normalized intensity value dictionary

        currentRound : int
            Current round number to calculate distances for

        currentRoundOmitNum : int
            Number of rounds that can be dropped from each barcode

        numJobs : int
            Number of CPU threads to use in parallel

    Returns
    -------
        pd.DataFrame : Modified spot table with added columns to with info on the "best" barcode
                       found for each spot
    '''

    # Calculate the spatial variance for each decodable barcode for each spot in each round
    if len(roundData) == 0:
        return roundData

    if 'targets' in roundData.columns:
        checkTargets = True
    else:
        checkTargets = False

    # Extract spot codes and targets
    allSpotCodes = [decodeSpots(codes, len(spotCoords)) for codes in roundData['spot_codes']]
    if checkTargets:
        allTargets = roundData['targets'].tolist()
    else:
        allTargets = [[0 for code in codes] for codes in roundData['spot_codes']]

    # Find ranges to chunk data by
    ranges = [0]
    for i in range(1, numJobs):
        ranges.append(int((len(roundData) / numJobs) * i))
    ranges.append(len(roundData))
    chunkedSpotCodes = [allSpotCodes[ranges[i]:ranges[i + 1]] for i in range(len(ranges[:-1]))]
    chunkedTargets = [allTargets[ranges[i]:ranges[i + 1]] for i in range(len(ranges[:-1]))]

    # Run in parallel
    with ProcessPoolExecutor() as pool:
        part = partial(distanceFunc, spotCoords=spotCoords, spotQualDict=spotQualDict,
                       currentRoundOmitNum=currentRoundOmitNum)
        poolMap = pool.map(part, [spotsAndTargets for spotsAndTargets in zip(chunkedSpotCodes,
                                                                             chunkedTargets)])
        results = [x for x in poolMap]

    # Add distances to decodedTables as new column and replace spot_codes and targets column with
    # only the min scoring values
    roundData['spot_codes'] = list(chain(*[job[0] for job in results]))
    roundData['distance'] = list(chain(*[job[1] for job in results]))
    if checkTargets:
        roundData['targets'] = list(chain(*[job[2] for job in results]))

    # Remove spots who had a tie between possible spot combinations
    roundData = roundData[roundData['spot_codes'] != -1]

    return roundData

def cleanup(bestPerSpotTables: dict,
            spotCoords: dict,
            channelDict: dict,
            strictness: int,
            currentRoundOmitNum: int,
            seedNumber: int) -> pd.DataFrame:

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

        channelDict : dict
            Dictionary with mapping between spot IDs and the channel labels

        strictness : int
            Parameter that determines how many possible barcodes each spot can have before it is
            dropped as ambiguous

        currentRoundOmitNum : int
            Number of rounds that can be dropped from each barcode

        seedNumber : A barcode must be chosen as "best" in this number of rounds to pass filters

    Returns
    -------
        pd.DataFrame : Dataframe containing final set of codes that have passed all filters
    '''

    # Create merged spot results dataframe containing the passing barcodes found in all the rounds
    mergedCodes = pd.DataFrame()
    roundNum = len(bestPerSpotTables)
    for r in range(roundNum):
        if len(bestPerSpotTables[r]) != 0:
            if strictness > 0:
                spotCodes = bestPerSpotTables[r]['spot_codes']
                targets = bestPerSpotTables[r]['targets']
                # Turn each barcode and spot code into a tuple so they can be used as dictionary
                # keys
                bestPerSpotTables[r]['spot_codes'] = [tuple(spotCode[0]) for spotCode in spotCodes]
                bestPerSpotTables[r]['targets'] = [target[0] for target in targets]
            mergedCodes = mergedCodes.append(bestPerSpotTables[r])
    mergedCodes = mergedCodes.reset_index(drop=True)

    # If no codes return empty dataframe
    if len(mergedCodes) == 0:
        return pd.DataFrame()

    # Only pass codes that are chosen as best for at least 2 of the spots that make it up
    spotCodes = mergedCodes['spot_codes']
    counts = defaultdict(int)  # type: dict
    for code in spotCodes:
        counts[code] += 1
    passing = list(set(code for code in counts if counts[code] >= seedNumber))

    passingCodes = mergedCodes[mergedCodes['spot_codes'].isin(passing)].reset_index(drop=True)
    passingCodes = passingCodes.iloc[passingCodes['spot_codes'].drop_duplicates().index]
    passingCodes = passingCodes.reset_index(drop=True)

    # If no codes return empty dataframe
    if len(passingCodes) == 0:
        return pd.DataFrame()

    # Need to find maximum independent set of spot codes where each spot code is a node and there
    # is an edge connecting two codes if they share at least one spot. Does this by eliminating
    # nodes (spot codes) that have the most edges first and if there is tie for which has the most
    # edges they are ordered in order of decreasing spatial variance of the spots that make it up
    # (so codes are eliminated in order first of how many other codes they share a spots with and
    # then spatial variance is used to break ties). Nodes are eliminated from the graph in this way
    # until there are no more edges in the graph

    # First prepare list of counters of the spot IDs for each round
    spotCodes = passingCodes['spot_codes']
    codeArray = np.asarray([np.asarray(code) for code in spotCodes])
    counters = []  # type: typing.List[Counter]
    for r in range(roundNum):
        counters.append(Counter(codeArray[:, r]))
        counters[-1][0] = 0

    # Then create collisonCounter dictionary which has the number of edges for each code and the
    # collisions dictionary which holds a list of codes each code has an overlap with. Any code with
    # no overlaps is added to keep to save later
    collisionCounter = defaultdict(int)  # type: dict
    collisions = defaultdict(list)
    keep = []
    for i, spotCode in enumerate(spotCodes):
        collision = False
        for r in range(roundNum):
            if spotCode[r] != 0:
                count = counters[r][spotCode[r]] - 1
                if count > 0:
                    collision = True
                    collisionCounter[spotCode] += count
                    collisions[spotCode].extend([spotCodes[ind[0]] for ind in
                                                 np.argwhere(codeArray[:, r] == spotCode[r])
                                                 if ind[0] != i])
        if not collision:
            keep.append(i)

    # spotDict dictionary has mapping for codes to their index location in spotCodes and
    # codeDistance has mapping for codes to their spatial variance value
    spotDict = {code: i for i, code in enumerate(spotCodes)}
    codeDistance = passingCodes.set_index('spot_codes')['distance'].to_dict()
    while len(collisions):
        # Gets all the codes that have the highest value for number of edges, and then sorts them by
        # their spatial variance values in decreasing order
        maxValue = max(collisionCounter.values())
        maxCodes = [code for code in collisionCounter if collisionCounter[code] == maxValue]
        distances = np.asarray([codeDistance[code] for code in maxCodes])
        sortOrder = [item[1] for item in sorted(zip(distances, range(len(distances))),
                                                reverse=True)]
        maxCodes = [tuple(code) for code in np.asarray(maxCodes)[sortOrder]]

        # For every maxCode, first check that it is still a maxCode (may change during this loop),
        # if it is then modify all the nodes that have edge to it to have one less edge (if this
        # causes that node to have no more edges then delete it from the graph and add it to the
        # codes we keep), then delete the maxCode from the graph
        for maxCode in maxCodes:
            if collisionCounter[maxCode] == maxValue:
                for code in collisions[maxCode]:
                    if collisionCounter[code] == 1:
                        del collisionCounter[code]
                        del collisions[code]
                        keep.append(spotDict[code])
                    else:
                        collisionCounter[code] -= 1
                        collisions[code] = [c for c in collisions[code] if c != maxCode]

                del collisionCounter[maxCode]
                del collisions[maxCode]

    # Only choose codes that we found to not have any edges in the graph
    finalCodes = passingCodes.loc[keep].reset_index(drop=True)

    if len(finalCodes) == 0:
        return pd.DataFrame()

    # Add barcode lables, spot coordinates, barcode center coordinates, and number of rounds used
    # for each barcode to table
    barcodes = []
    allCoords = []
    centers = []
    roundsUsed = []
    # intensities = []
    for i in range(len(finalCodes)):
        spotCode = finalCodes.iloc[i]['spot_codes']
        barcodes.append([channelDict[j][spot] for j, spot in enumerate(spotCode)])
        counter = Counter(spotCode)  # type: Counter
        roundsUsed.append(roundNum - counter[0])
        coords = np.asarray([spotCoords[j][spot] for j, spot in enumerate(spotCode) if spot != 0])
        allCoords.append(coords)
        coords = np.asarray([coord for coord in coords])
        center = np.asarray(coords).mean(axis=0)
        centers.append(tuple(center))
        # intensities.append([spotIntensities[j][spot] for j,spot in enumerate(spotCode)])
    finalCodes['best_barcodes'] = barcodes
    finalCodes['coords'] = allCoords
    finalCodes['center'] = centers
    # finalCodes['intensities'] = intensities
    finalCodes['rounds_used'] = roundsUsed

    return finalCodes

def removeUsedSpots(finalCodes: pd.DataFrame, spotTables: dict) -> dict:

    '''
    Remove spots found to be in barcodes for the current round omission number from the spotTables
    so they are not used for the next round omission number

    Parameters
    ----------
        finalCodes : pd.DataFrame
            Dataframe containing final set of codes that have passed all filters

        spotTables : dict
            Dictionary of original data tables extracted from SpotFindingResults objects by the
            _merge_spots_by_round() function

    Returns
    -------
        dict : Modified version of spotTables with spots that have been used in the current round
               omission removed
    '''

    # Remove used spots
    for r in range(len(spotTables)):
        usedSpots = set([passed[r] for passed in finalCodes['spot_codes']
                         if passed[r] != 0])
        spotTables[r] = spotTables[r][~spotTables[r]['spot_id'].isin(usedSpots)]
        spotTables[r] = spotTables[r].reset_index(drop=True)
        spotTables[r].index = range(1, len(spotTables[r]) + 1)

    return spotTables
