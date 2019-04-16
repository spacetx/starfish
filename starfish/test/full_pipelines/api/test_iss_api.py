import os
import sys
import tempfile

import numpy as np

import starfish
from starfish import IntensityTable
from starfish.spots import TargetAssignment
from starfish.types import Coordinates, Features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(starfish.__file__)))
os.environ["USE_TEST_DATA"] = "1"
sys.path.append(os.path.join(ROOT_DIR, "notebooks", "py"))


def test_iss_pipeline_cropped_data():

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
        [[9.989060e-03, 4.386014e-03, 3.409548e-03, 1.683664e-03, 1.789603e-04,
          0.000000e+00, 1.040482e-04, 1.472828e-05, 1.062776e-03, 6.548513e-03],
         [1.452212e-02, 9.637723e-03, 8.215415e-03, 5.938874e-03, 1.818186e-03,
          3.514882e-04, 5.291665e-04, 3.798662e-04, 4.371551e-05, 1.081357e-03],
         [2.307975e-02, 1.585000e-02, 1.239828e-02, 9.505798e-03, 3.524747e-03,
          1.483906e-03, 1.330130e-03, 2.842413e-04, 5.669349e-04, 3.735562e-03],
         [3.528022e-02, 2.443082e-02, 1.964650e-02, 1.254200e-02, 7.762027e-03,
          4.916940e-03, 2.758998e-03, 3.099556e-04, 4.916890e-04, 5.299725e-03],
         [5.145832e-02, 3.791203e-02, 3.143204e-02, 2.311396e-02, 1.554033e-02,
          9.379421e-03, 6.131854e-03, 7.163712e-04, 1.246572e-03, 2.651690e-03],
         [5.950670e-02, 5.169559e-02, 4.459712e-02, 3.412621e-02, 2.399661e-02,
          1.657831e-02, 1.188810e-02, 4.341024e-03, 1.799579e-03, 1.730873e-03],
         [5.873162e-02, 5.882759e-02, 5.405201e-02, 4.136665e-02, 3.178160e-02,
          2.466743e-02, 1.444380e-02, 6.790231e-03, 6.046029e-03, 2.565000e-03],
         [4.810252e-02, 5.582267e-02, 5.533365e-02, 4.697119e-02, 3.491036e-02,
          2.578102e-02, 1.868390e-02, 1.031733e-02, 8.699552e-03, 2.899562e-03],
         [4.105438e-02, 4.542167e-02, 4.910203e-02, 4.965963e-02, 4.018034e-02,
          2.968885e-02, 1.948875e-02, 1.384566e-02, 8.774363e-03, 6.954662e-03],
         [3.554880e-02, 3.779816e-02, 4.067500e-02, 4.671254e-02, 4.536573e-02,
          3.356919e-02, 2.238698e-02, 1.681009e-02, 1.111025e-02, 1.012993e-02]],
        dtype=np.float32
    )

    assert np.allclose(
        expected_registered_values,
        registered_image.xarray[2, 2, 0, 40:50, 40:50]
    )

    pipeline_log = registered_image.log

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'
    assert pipeline_log[3]['method'] == 'BlobDetector'

    intensities = iss.intensities

    # assert that the number of spots detected is 99
    assert intensities.sizes[Features.AXIS] == 99

    # decode
    decoded = iss.decoded

    # decoding identifies 4 genes, each with 1 count
    genes, gene_counts = iss.genes, iss.counts
    assert np.array_equal(genes, np.array(['ACTB', 'CD68', 'CTSL2', 'EPCAM',
                                           'ETV4', 'GAPDH', 'GUS', 'HER2', 'RAC1',
                                           'TFRC', 'TP53', 'VEGF']))
    assert np.array_equal(gene_counts, [20, 1, 5, 2, 1, 11, 1, 3, 2, 1, 1, 2])

    masks = iss.masks

    seg = iss.seg

    # segmentation identifies only one cell
    assert seg._segmentation_instance.num_cells == 1

    # assign targets
    lab = TargetAssignment.Label()
    assigned = lab.run(masks, decoded)

    pipeline_log = assigned.get_log()

    # assert tht physical coordinates were transferred
    assert Coordinates.X in assigned.coords
    assert Coordinates.Y in assigned.coords
    assert Coordinates.Z in assigned.coords

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'
    assert pipeline_log[3]['method'] == 'BlobDetector'

    # Test serialization / deserialization of IntensityTable log
    fp = tempfile.NamedTemporaryFile()
    assigned.save(fp.name)
    loaded_intensities = IntensityTable.load(fp.name)
    pipeline_log = loaded_intensities.get_log()

    assert pipeline_log[0]['method'] == 'WhiteTophat'
    assert pipeline_log[1]['method'] == 'Warp'
    assert pipeline_log[3]['method'] == 'BlobDetector'

    # 28 of the spots are assigned to cell 1 (although most spots do not decode!)
    assert np.sum(assigned['cell_id'] == '1') == 28
