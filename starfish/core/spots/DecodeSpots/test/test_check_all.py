import random

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from starfish import ImageStack
from starfish.core.codebook.codebook import Codebook
from starfish.core.spots.DecodeSpots.check_all_decoder import CheckAll
from starfish.core.spots.FindSpots import BlobDetector

def syntheticSeqfish(x, y, z, codebook, nSpots, jitter, error):
    nRound = codebook.shape[1]
    nChannel = codebook.shape[2]
    img = np.zeros((nRound, nChannel, z, y, x), dtype=np.float32)

    intCodes = np.argmax(codebook.data, axis=2)

    targets = []
    for _ in range(nSpots):
        randx = random.choice(range(5, x - 5))
        randy = random.choice(range(5, y - 5))
        randz = random.choice(range(2, z - 2))
        randCode = random.choice(range(len(codebook)))
        targets.append((randCode, (randx, randy, randz)))
        if jitter > 0:
            randx += random.choice(range(jitter + 1)) * random.choice([1, -1])
            randy += random.choice(range(jitter + 1)) * random.choice([1, -1])
        if error:
            skip = random.choice(range(nRound))
        else:
            skip = 100
        for r, ch in enumerate(intCodes[randCode]):
            if r != skip:
                img[r, ch, randz, randy, randx] = 10

    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img / img.max()), targets


def seqfishCodebook(nRound, nChannel, nCodes):

    def barcodeConv(lis, chs):
        barcode = np.zeros((len(lis), chs))
        for i in range(len(lis)):
            barcode[i][lis[i]] = 1
        return barcode

    def incrBarcode(lis, chs):
        currInd = len(lis) - 1
        lis[currInd] += 1
        while lis[currInd] == chs:
            lis[currInd] = 0
            currInd -= 1
            lis[currInd] += 1
        return lis

    allCombo = np.zeros((nChannel ** nRound, nRound, nChannel))

    barcode = [0] * nRound
    for i in range(np.shape(allCombo)[0]):
        allCombo[i] = barcodeConv(barcode, nChannel)
        barcode = incrBarcode(barcode, nChannel)

    hammingDistance = 1
    blanks = []
    i = 0
    while i < len(allCombo):
        blanks.append(allCombo[i])
        j = i + 1
        while j < len(allCombo):
            if np.count_nonzero(~(allCombo[i] == allCombo[j])) / 2 <= hammingDistance:
                allCombo = allCombo[[k for k in range(len(allCombo)) if k != j]]
            else:
                j += 1
        i += 1

    data = np.asarray(blanks)[random.sample(range(len(blanks)), nCodes)]

    return Codebook.from_numpy(code_names=range(len(data)), n_round=nRound,
                               n_channel=nChannel, data=data)

def testExactMatches():

    codebook = seqfishCodebook(5, 3, 20)

    img, trueTargets = syntheticSeqfish(100, 100, 20, codebook, 20, 0, False)

    bd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=30, threshold=.1, exclude_border=False)
    spots = bd.run(image_stack=img)
    assert spots.count_total_spots() == 5 * 20, 'Spot detector did not find all spots'

    decoder = CheckAll(codebook=codebook, search_radius=1, error_rounds=0)
    hits = decoder.run(spots=spots, n_processes=4)

    testTargets = []
    for i in range(len(hits)):
        testTargets.append((int(hits[i]['target'].data),
                           (int(hits[i]['x'].data), int(hits[i]['y'].data),
                            int(hits[i]['z'].data))))

    matches = 0
    for true in trueTargets:
        for test in testTargets:
            if true[0] == test[0]:
                if test[1][0] + 1 >= true[1][0] >= test[1][0] - 1 and \
                   test[1][1] + 1 >= true[1][1] >= test[1][1] - 1:
                    matches += 1

    assert matches == len(trueTargets)

def testJitteredMatches():

    codebook = seqfishCodebook(5, 3, 20)

    img, trueTargets = syntheticSeqfish(100, 100, 20, codebook, 20, 2, False)

    bd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=30, threshold=.1, exclude_border=False)
    spots = bd.run(image_stack=img)
    assert spots.count_total_spots() == 5 * 20, 'Spot detector did not find all spots'

    decoder = CheckAll(codebook=codebook, search_radius=3, error_rounds=0)
    hits = decoder.run(spots=spots, n_processes=4)

    testTargets = []
    for i in range(len(hits)):
        testTargets.append((int(hits[i]['target'].data),
                           (int(hits[i]['x'].data), int(hits[i]['y'].data),
                            int(hits[i]['z'].data))))

    matches = 0
    for true in trueTargets:
        for test in testTargets:
            if true[0] == test[0]:
                if test[1][0] + 3 >= true[1][0] >= test[1][0] - 3 and \
                   test[1][1] + 3 >= true[1][1] >= test[1][1] - 3:
                    matches += 1

    assert matches == len(trueTargets)

def testErrorCorrection():

    codebook = seqfishCodebook(5, 3, 20)

    img, trueTargets = syntheticSeqfish(100, 100, 20, codebook, 20, 0, True)

    bd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=30, threshold=.1, exclude_border=False)
    spots = bd.run(image_stack=img)
    assert spots.count_total_spots() == 4 * 20, 'Spot detector did not find all spots'

    decoder = CheckAll(codebook=codebook, search_radius=1, error_rounds=1)
    hits = decoder.run(spots=spots, n_processes=4)

    testTargets = []
    for i in range(len(hits)):
        testTargets.append((int(str(hits[i]['target'].data).split('.')[0]),
                           (int(hits[i]['x'].data), int(hits[i]['y'].data),
                            int(hits[i]['z'].data))))

    matches = 0
    for true in trueTargets:
        for test in testTargets:
            if true[0] == test[0]:
                if test[1][0] + 1 >= true[1][0] >= test[1][0] - 1 and \
                   test[1][1] + 1 >= true[1][1] >= test[1][1] - 1:
                    matches += 1

    assert matches == len(trueTargets)
