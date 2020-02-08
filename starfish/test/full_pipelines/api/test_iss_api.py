import os
import sys
import tempfile

import numpy as np

import starfish
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.spots import AssignTargets
from starfish.types import Coordinates, Features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(starfish.__file__)))
os.environ["USE_TEST_DATA"] = "1"
sys.path.append(os.path.join(ROOT_DIR, "notebooks", "py"))


def test_iss_pipeline_cropped_data(tmpdir):

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    iss = __import__('ISS')

    white_top_hat_filtered_image = iss.filtered_imgs

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

    registered_image = iss.registered_imgs

    expected_registered_values = np.array(
        [[9.972601e-03, 4.410370e-03, 3.392192e-03, 1.687834e-03, 1.880155e-04,
          0.000000e+00, 1.047019e-04, 1.578360e-05, 1.069453e-03, 6.543968e-03],
         [1.456979e-02, 9.646147e-03, 8.203185e-03, 5.936079e-03, 1.839891e-03,
          3.569032e-04, 5.237808e-04, 3.792955e-04, 4.592746e-05, 1.088151e-03],
         [2.313178e-02, 1.586836e-02, 1.240375e-02, 9.513815e-03, 3.563545e-03,
          1.488329e-03, 1.326624e-03, 2.939297e-04, 5.607218e-04, 3.690171e-03],
         [3.531289e-02, 2.446796e-02, 1.964004e-02, 1.258251e-02, 7.771713e-03,
          4.918387e-03, 2.766922e-03, 3.267574e-04, 4.892451e-04, 5.261183e-03],
         [5.146676e-02, 3.794888e-02, 3.141785e-02, 2.312119e-02, 1.555709e-02,
          9.402979e-03, 6.135746e-03, 7.547007e-04, 1.231891e-03, 2.656648e-03],
         [5.952225e-02, 5.170041e-02, 4.459279e-02, 3.416265e-02, 2.403326e-02,
          1.659481e-02, 1.189285e-02, 4.377660e-03, 1.810592e-03, 1.729033e-03],
         [5.872828e-02, 5.881007e-02, 5.405803e-02, 4.143796e-02, 3.181438e-02,
          2.468321e-02, 1.451422e-02, 6.834699e-03, 6.021897e-03, 2.588449e-03],
         [4.815195e-02, 5.578594e-02, 5.535153e-02, 4.701486e-02, 3.499170e-02,
          2.584777e-02, 1.871042e-02, 1.036013e-02, 8.698075e-03, 2.945077e-03],
         [4.108098e-02, 4.543370e-02, 4.911040e-02, 4.965232e-02, 4.022935e-02,
          2.973786e-02, 1.956365e-02, 1.386791e-02, 8.811617e-03, 6.941982e-03],
         [3.560406e-02, 3.779930e-02, 4.068928e-02, 4.668610e-02, 4.536487e-02,
          3.364870e-02, 2.244582e-02, 1.683235e-02, 1.113740e-02, 1.012298e-02]],
        dtype=np.float32
    )

    assert np.allclose(
        expected_registered_values,
        registered_image.xarray[2, 2, 0, 40:50, 40:50]
    )

    pipeline_log = registered_image.log.data

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'

    # decode
    decoded = iss.decoded

    # decoding identifies 4 genes, each with 1 count
    genes, gene_counts = iss.genes, iss.counts
    assert np.array_equal(genes, np.array(['ACTB', 'CD68', 'CTSL2', 'EPCAM',
                                           'ETV4', 'GAPDH', 'GUS', 'HER2', 'RAC1',
                                           'TFRC', 'TP53', 'VEGF']))

    assert np.array_equal(gene_counts, [19, 1, 5, 2, 1, 11, 1, 3, 2, 1, 1, 2])
    assert decoded.sizes[Features.AXIS] == 99

    masks = iss.masks

    # segmentation identifies only one cell
    assert len(iss.watershed_markers) == 6

    # assign targets
    lab = AssignTargets.Label()
    assigned = lab.run(masks, decoded)

    pipeline_log = assigned.get_log()

    # assert tht physical coordinates were transferred
    assert Coordinates.X in assigned.coords
    assert Coordinates.Y in assigned.coords
    assert Coordinates.Z in assigned.coords

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'
    assert pipeline_log[2]['method'] == 'BlobDetector'
    assert pipeline_log[3]['method'] == 'PerRoundMaxChannel'

    # Test serialization / deserialization of IntensityTable log
    with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False) as ntf:
        tfp = ntf.name
    assigned.to_netcdf(tfp)
    loaded_intensities = IntensityTable.open_netcdf(tfp)
    pipeline_log = loaded_intensities.get_log()

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'
    assert pipeline_log[2]['method'] == 'BlobDetector'
    assert pipeline_log[3]['method'] == 'PerRoundMaxChannel'

    # 28 of the spots are assigned to cell 0 (although most spots do not decode!)
    assert np.sum(assigned['cell_id'] == '1') == 28

    expression_matrix = iss.cg
    # test that nans were properly removed from the expression matrix
    assert 'nan' not in expression_matrix.genes.data
    # test the number of spots that did not decode per cell
    assert np.array_equal(expression_matrix.number_of_undecoded_spots.data, [13, 1, 0, 36])
