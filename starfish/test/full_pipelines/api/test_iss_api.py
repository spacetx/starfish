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
        [[9.956287e-03, 4.382827e-03, 3.377583e-03, 1.676941e-03, 1.843710e-04,
          0.000000e+00, 1.060593e-04, 1.570793e-05, 1.083317e-03, 6.584799e-03],
         [1.453807e-02, 9.621573e-03, 8.186622e-03, 5.916186e-03, 1.824629e-03,
          3.533170e-04, 5.222907e-04, 3.772743e-04, 4.503722e-05, 1.102258e-03],
         [2.307469e-02, 1.583596e-02, 1.238587e-02, 9.499015e-03, 3.546112e-03,
          1.481345e-03, 1.324189e-03, 2.925306e-04, 5.596119e-04, 3.681781e-03],
         [3.524983e-02, 2.441847e-02, 1.960656e-02, 1.255823e-02, 7.747118e-03,
          4.900453e-03, 2.757738e-03, 3.212543e-04, 4.914876e-04, 5.271485e-03],
         [5.139222e-02, 3.787590e-02, 3.136785e-02, 2.306333e-02, 1.551601e-02,
          9.377318e-03, 6.117928e-03, 7.416179e-04, 1.231178e-03, 2.670616e-03],
         [5.949953e-02, 5.164410e-02, 4.453927e-02, 3.411048e-02, 2.398818e-02,
          1.655605e-02, 1.186486e-02, 4.347167e-03, 1.806095e-03, 1.728893e-03],
         [5.874086e-02, 5.879546e-02, 5.402317e-02, 4.139205e-02, 3.177369e-02,
          2.464539e-02, 1.448956e-02, 6.815665e-03, 6.004986e-03, 2.577132e-03],
         [4.819431e-02, 5.582142e-02, 5.535191e-02, 4.698015e-02, 3.496249e-02,
          2.583509e-02, 1.868034e-02, 1.033007e-02, 8.692421e-03, 2.931876e-03],
         [4.109203e-02, 4.548026e-02, 4.914177e-02, 4.964774e-02, 4.018743e-02,
          2.969893e-02, 1.954622e-02, 1.384825e-02, 8.802789e-03, 6.922520e-03],
         [3.561261e-02, 3.782343e-02, 4.072055e-02, 4.671327e-02, 4.534628e-02,
          3.361325e-02, 2.241083e-02, 1.681411e-02, 1.111553e-02, 1.011566e-02]],
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

    # segmentation identifies four cells
    assert len(iss.watershed_markers) == 4

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
    assert np.array_equal(expression_matrix.number_of_undecoded_spots.data, [13, 1, 36])
