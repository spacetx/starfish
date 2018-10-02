import os
import sys

import numpy as np

import starfish
from starfish.spots._target_assignment.point_in_poly import PointInPoly2D
from starfish.types import Features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(starfish.__file__)))
os.environ["USE_TEST_DATA"] = "1"
sys.path.append(os.path.join(ROOT_DIR, "notebooks", "py"))


def test_iss_pipeline_cropped_data():

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    iss = __import__('ISS_Pipeline_-_Breast_-_1_FOV')

    white_top_hat_filtered_image = iss.primary_image

    # # pick a random part of the registered image and assert on it
    expected_filtered_values = np.array(
        [[0.1041123, 0.09968718, 0.09358358, 0.09781034, 0.08943313, 0.08853284,
          0.08714428, 0.07518119, 0.07139697, 0.0585336, ],
         [0.09318685, 0.09127947, 0.0890364, 0.094728, 0.08799877, 0.08693064,
          0.08230717, 0.06738383, 0.05857938, 0.04223698],
         [0.08331426, 0.0812543, 0.08534371, 0.0894789, 0.09184404, 0.08274967,
          0.0732433, 0.05564965, 0.04577706, 0.03411917],
         [0.06741435, 0.07370108, 0.06511024, 0.07193103, 0.07333485, 0.07672236,
          0.06019684, 0.04415961, 0.03649958, 0.02737468],
         [0.05780118, 0.06402685, 0.05947966, 0.05598535, 0.05192646, 0.04870679,
          0.04164187, 0.03291371, 0.03030441, 0.02694743],
         [0.04649424, 0.06117342, 0.05899138, 0.05101091, 0.03639277, 0.03379873,
          0.03382925, 0.0282597, 0.02383459, 0.01651026],
         [0.0414435, 0.04603647, 0.05458152, 0.04969863, 0.03799496, 0.0325475,
          0.02928206, 0.02685588, 0.02172885, 0.01722743],
         [0.04107728, 0.04161135, 0.04798963, 0.05156023, 0.03952087, 0.02899214,
          0.02589456, 0.02824444, 0.01815823, 0.01557945],
         [0.03901731, 0.03302052, 0.03498893, 0.03929199, 0.03695735, 0.02943466,
          0.01945525, 0.01869231, 0.01666284, 0.01240558],
         [0.02664226, 0.02386511, 0.02206454, 0.02978561, 0.03265431, 0.0265507,
          0.02214084, 0.01844815, 0.01542687, 0.01353475]],
        dtype=np.float32
    )

    assert white_top_hat_filtered_image.xarray.dtype == np.float32

    assert np.allclose(
        expected_filtered_values,
        white_top_hat_filtered_image.xarray[2, 2, 0, 40:50, 40:50]
    )

    registered_image = iss.registered_image

    # assert on a random part of the filtered image
    expected_registered_values = np.array(
        [[1.15684755e-02, 3.86373512e-03, 2.60785664e-03, 1.35898031e-03,
          0.00000000e+00, 0.00000000e+00, 1.15468545e-04, 0.00000000e+00,
          0.00000000e+00, 7.16284662e-03],
         [1.33818155e-02, 8.74291547e-03, 8.54650512e-03, 5.62853599e-03,
          1.74340000e-03, 5.84440822e-05, 5.53897815e-04, 7.40510353e-04,
          1.09384397e-04, 0.00000000e+00],
         [2.32197642e-02, 1.48485349e-02, 1.20572122e-02, 1.00518325e-02,
          3.78616550e-03, 5.51953330e-04, 1.58034940e-03, 0.00000000e+00,
          6.80672762e-04, 3.19095445e-03],
         [3.35171446e-02, 2.30573826e-02, 1.86911281e-02, 1.16252769e-02,
          6.42230362e-03, 4.63001803e-03, 2.50486028e-03, 1.08768849e-03,
          0.00000000e+00, 6.75229309e-03],
         [5.30732870e-02, 3.61515097e-02, 3.07820514e-02, 2.19761301e-02,
          1.52691351e-02, 9.15373303e-03, 5.41988341e-03, 0.00000000e+00,
          9.24524793e-04, 3.14691127e-03],
         [6.05984218e-02, 5.18058091e-02, 4.45022509e-02, 3.48668806e-02,
          2.37355866e-02, 1.48193939e-02, 1.27216997e-02, 4.51163156e-03,
          9.03905951e-04, 1.46147527e-03],
         [6.03375509e-02, 6.00600466e-02, 5.51640466e-02, 4.15391065e-02,
          3.14524844e-02, 2.64199078e-02, 1.41928447e-02, 6.39168778e-03,
          5.34856878e-03, 3.28401709e-03],
         [4.75437082e-02, 5.76229692e-02, 5.73692545e-02, 4.73626107e-02,
          3.53275351e-02, 2.55322661e-02, 1.92126688e-02, 9.48204752e-03,
          9.59323533e-03, 3.06630856e-03],
         [3.94404493e-02, 4.60249558e-02, 4.95812669e-02, 5.13879694e-02,
          3.92197557e-02, 2.88589876e-02, 1.97741222e-02, 1.41243441e-02,
          8.03299714e-03, 6.66760467e-03],
         [3.60199437e-02, 3.75546440e-02, 4.03789952e-02, 4.74576391e-02,
          4.75049056e-02, 3.51673886e-02, 2.08846033e-02, 1.69556923e-02,
          1.06233349e-02, 1.02646966e-02]],
        dtype=np.float32
    )
    assert np.allclose(
        expected_registered_values,
        registered_image.xarray[2, 2, 0, 40:50, 40:50]
    )

    intensities = iss.intensities

    # assert that the number of spots detected is 99
    assert intensities.sizes[Features.AXIS] == 99

    # decode
    decoded = iss.decoded

    # decoding identifies 4 genes, each with 1 count
    genes, gene_counts = iss.genes, iss.counts
    assert np.array_equal(genes, np.array(['ACTB', 'CD68', 'CTSL2', 'EPCAM', 'ETV4', 'GAPDH',
                                           'HER2', 'MET', 'RAC1', 'TFRC', 'TP53', 'VEGF']))
    assert np.array_equal(gene_counts, [18, 1, 5, 2, 1, 12, 3, 1, 2, 1, 1, 2])

    regions = iss.regions

    seg = iss.seg

    # segmentation identifies only one cell
    assert seg._segmentation_instance.num_cells == 1

    # assign targets
    pip = PointInPoly2D()
    pip.run(decoded, regions)

    # 18 of the spots are assigned to cell 1 (although most spots do not decode!)
    assert np.sum(decoded['cell_id'] == 0) == 18
