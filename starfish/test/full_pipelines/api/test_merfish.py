import numpy as np
import pandas as pd

from starfish import Codebook, Experiment
from starfish.image._filter.gaussian_high_pass import GaussianHighPass
from starfish.image._filter.gaussian_low_pass import GaussianLowPass
from starfish.image._filter.richardson_lucy_deconvolution import DeconvolvePSF
from starfish.spots._detector.pixel_spot_detector import PixelSpotDetector
from starfish.types import Features, Indices


def test_merfish_pipeline_cropped_data():

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    # load the experiment
    experiment_json = (
        "https://dmf0bdeheu4zf.cloudfront.net/20180828/MERFISH-TEST/experiment.json"
    )
    experiment = Experiment.from_json(experiment_json)
    primary_image = experiment.fov().primary_image

    expected_primary_image = np.array(
        [[6287, 6419, 6612, 6705, 6641, 6555, 6784, 6978, 7084, 7058],
         [6449, 6364, 6414, 6570, 6565, 6621, 7049, 7178, 7136, 6863],
         [6531, 6553, 6755, 6727, 6665, 6827, 6934, 6985, 6864, 6692],
         [6805, 6895, 6962, 6898, 6816, 6629, 6687, 6826, 6781, 6838],
         [6907, 6913, 6858, 6912, 6957, 6917, 6768, 6547, 6541, 6751],
         [7111, 6979, 6778, 6820, 7059, 7039, 6986, 6768, 6730, 6715],
         [7489, 7354, 7181, 7035, 7013, 7066, 7045, 6781, 6652, 6765],
         [7630, 7498, 7322, 7199, 7234, 7194, 7116, 6735, 6718, 6667],
         [7620, 7627, 7590, 7516, 7332, 7237, 7205, 6877, 6688, 6619],
         [7673, 7728, 7626, 7696, 7461, 7312, 7156, 6990, 6770, 6662]],
        dtype=np.uint16
    )
    assert np.array_equal(
        expected_primary_image,
        primary_image.numpy_array[5, 0, 0, 40:50, 45:55]
    )

    # high pass filter
    ghp = GaussianHighPass(sigma=3)
    high_passed = ghp.run(primary_image, in_place=False)

    expected_high_passed = np.array(
        [[0, 0, 72, 131, 39, 0, 134, 306, 388, 337],
         [0, 0, 0, 0, 0, 0, 332, 443, 381, 86],
         [0, 0, 27, 0, 0, 63, 161, 202, 68, 0],
         [0, 61, 127, 65, 0, 0, 0, 5, 0, 5],
         [0, 0, 0, 0, 48, 28, 0, 0, 0, 0],
         [0, 0, 0, 0, 69, 87, 71, 0, 0, 0],
         [216, 128, 5, 0, 0, 49, 81, 0, 0, 0],
         [233, 159, 44, 0, 86, 114, 103, 0, 0, 0],
         [123, 195, 226, 225, 117, 100, 144, 0, 0, 0],
         [100, 224, 196, 344, 191, 126, 51, 0, 0, 0]],
        dtype=np.uint16
    )

    assert np.array_equal(
        expected_high_passed,
        high_passed.numpy_array[5, 0, 0, 40:50, 45:55]
    )

    # deconvolve the point spread function
    dpsf = DeconvolvePSF(num_iter=15, sigma=2)
    deconvolved = dpsf.run(high_passed, in_place=False)

    # assert that the deconvolved data is correct
    expected_deconvolved_values = np.array(
        [[0, 0, 3, 15, 28, 55, 152, 323, 425, 364],
         [0, 0, 1, 7, 23, 74, 259, 508, 506, 284],
         [0, 0, 1, 7, 20, 47, 105, 117, 72, 27],
         [0, 0, 2, 7, 13, 16, 15, 6, 1, 0],
         [0, 1, 3, 6, 8, 6, 2, 0, 0, 0],
         [5, 4, 5, 6, 7, 3, 0, 0, 0, 0],
         [48, 20, 14, 12, 11, 4, 0, 0, 0, 0],
         [169, 77, 59, 52, 46, 16, 1, 0, 0, 0],
         [125, 98, 142, 202, 220, 77, 5, 0, 0, 0],
         [30, 61, 227, 619, 890, 315, 17, 0, 0, 0]],
        dtype=np.uint16
    )
    assert np.array_equal(
        expected_deconvolved_values,
        deconvolved.numpy_array[5, 0, 0, 40:50, 45:55]
    )

    # low pass filter
    glp = GaussianLowPass(sigma=1)
    low_passed = glp.run(deconvolved, in_place=False)

    expected_low_passed = np.array(
        [[2, 9, 19, 30, 43, 77, 149, 228, 260, 237],
         [0, 1, 6, 15, 37, 87, 175, 258, 274, 226],
         [0, 0, 3, 10, 27, 61, 113, 153, 149, 112],
         [0, 1, 3, 8, 15, 27, 42, 49, 42, 28],
         [6, 4, 4, 6, 9, 10, 10, 9, 6, 3],
         [33, 16, 11, 10, 9, 6, 3, 1, 0, 0],
         [95, 49, 34, 31, 26, 15, 5, 1, 0, 0],
         [151, 91, 85, 98, 92, 55, 19, 3, 0, 0],
         [134, 114, 163, 243, 258, 164, 58, 11, 1, 0],
         [73, 102, 218, 395, 461, 309, 113, 21, 2, 0]],
        dtype=np.uint16
    )
    assert np.array_equal(
        expected_low_passed,
        low_passed.numpy_array[5, 0, 0, 40:50, 45:55]
    )

    # cast to float
    low_passed._data = low_passed._data.astype(np.float64)

    # scale the data by the scale factors
    scale_factors = {
        (t[Indices.ROUND], t[Indices.CH]): t['scale_factor']
        for t in experiment.extras['scale_factors']
    }
    for indices in low_passed._iter_indices():
        data = low_passed.get_slice(indices)[0]
        scaled = data / scale_factors[indices[Indices.ROUND.value], indices[Indices.CH.value]]
        low_passed.set_slice(indices, scaled)

    # assert that the scaled data is correct
    expected_scaled_low_passed = np.array(
        [[0.02516705, 0.11325171, 0.23908694, 0.37750569, 0.54109149,
          0.96893128, 1.87494495, 2.86904327, 3.27171602, 2.98229498],
         [0., 0.01258352, 0.07550114, 0.18875285, 0.46559036,
          1.09476651, 2.20211655, 3.24654897, 3.44788534, 2.84387623],
         [0., 0., 0.03775057, 0.12583523, 0.33975512,
          0.76759491, 1.42193811, 1.92527904, 1.87494495, 1.40935459],
         [0., 0.01258352, 0.03775057, 0.10066819, 0.18875285,
          0.33975512, 0.52850797, 0.61659263, 0.52850797, 0.35233865],
         [0.07550114, 0.05033409, 0.05033409, 0.07550114, 0.11325171,
          0.12583523, 0.12583523, 0.11325171, 0.07550114, 0.03775057],
         [0.41525626, 0.20133637, 0.13841875, 0.12583523, 0.11325171,
          0.07550114, 0.03775057, 0.01258352, 0., 0.],
         [1.1954347, 0.61659263, 0.42783979, 0.39008922, 0.3271716,
          0.18875285, 0.06291762, 0.01258352, 0., 0.],
         [1.90011199, 1.14510061, 1.06959947, 1.23318527, 1.15768413,
          0.69209377, 0.23908694, 0.03775057, 0., 0.],
         [1.6861921, 1.43452164, 2.05111427, 3.05779612, 3.24654897,
          2.06369779, 0.72984434, 0.13841875, 0.01258352, 0.],
         [0.91859719, 1.28351936, 2.74320804, 4.97049164, 5.80100417,
          3.88830865, 1.42193811, 0.26425399, 0.02516705, 0.]],
        dtype=np.float
    )
    assert np.allclose(
        expected_scaled_low_passed,
        low_passed.numpy_array[5, 0, 0, 40:50, 45:55]
    )

    # detect and decode spots
    psd = PixelSpotDetector(
        codebook=experiment.codebook,
        metric='euclidean',
        distance_threshold=0.5176,
        magnitude_threshold=1,
        min_area=2,
        max_area=np.inf,
        norm_order=2,
        crop_size=(0, 40, 40)
    )
    spot_intensities, prop_results = psd.run(low_passed)

    # verify that the number of spots are correct
    assert spot_intensities.sizes[Features.AXIS] == 1019

    # compare to paper results
    bench = pd.read_csv('https://dmf0bdeheu4zf.cloudfront.net/MERFISH/benchmark_results.csv',
                        dtype={'barcode': object})

    benchmark_counts = bench.groupby('gene')['gene'].count()
    genes, counts = np.unique(spot_intensities[Features.TARGET], return_counts=True)
    result_counts = pd.Series(counts, index=genes)

    # assert that number of high-expression detected genes are correct
    expected_counts = pd.Series(
        [101, 84, 70, 48, 40],
        index=(None, 'SRRM2', 'FASN', 'IGF2R', 'MYH10')
    )
    assert np.array_equal(
        expected_counts.values,
        result_counts.sort_values(ascending=False).head(5).values
    )

    tmp = pd.concat([result_counts, benchmark_counts], join='inner', axis=1).values

    corrcoef = np.corrcoef(tmp[:, 1], tmp[:, 0])[0, 1]

    assert np.round(corrcoef, 4) == 0.9166
