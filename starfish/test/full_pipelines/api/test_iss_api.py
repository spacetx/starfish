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

    expected_registered_values = np.array(
        [[9.712600e-03, 4.279962e-03, 3.146056e-03, 1.600749e-03, 2.020520e-04,
          0.000000e+00, 1.213742e-04, 2.175575e-05, 1.239748e-03, 6.913641e-03],
         [1.451844e-02, 9.455775e-03, 7.966662e-03, 5.723346e-03, 1.810986e-03,
          3.510148e-04, 4.782984e-04, 3.555592e-04, 4.935622e-05, 1.261424e-03],
         [2.286866e-02, 1.564934e-02, 1.225544e-02, 9.413136e-03, 3.614988e-03,
          1.445069e-03, 1.281242e-03, 3.348967e-04, 5.153420e-04, 3.349161e-03],
         [3.488191e-02, 2.418881e-02, 1.927061e-02, 1.257003e-02, 7.589337e-03,
          4.747066e-03, 2.718625e-03, 3.684438e-04, 4.966246e-04, 5.139431e-03],
         [5.077895e-02, 3.743769e-02, 3.083774e-02, 2.259170e-02, 1.524086e-02,
          9.280980e-03, 5.982058e-03, 8.432306e-04, 1.141085e-03, 2.816019e-03],
         [5.939967e-02, 5.118315e-02, 4.403604e-02, 3.385423e-02, 2.379876e-02,
          1.630748e-02, 1.164083e-02, 4.287360e-03, 1.823670e-03, 1.717638e-03],
         [5.884148e-02, 5.857381e-02, 5.375163e-02, 4.139594e-02, 3.160535e-02,
          2.440129e-02, 1.466982e-02, 6.907708e-03, 5.720033e-03, 2.605121e-03],
         [4.885358e-02, 5.593539e-02, 5.546480e-02, 4.693525e-02, 3.516930e-02,
          2.610784e-02, 1.857848e-02, 1.030868e-02, 8.629656e-03, 3.078087e-03],
         [4.134395e-02, 4.595317e-02, 4.947726e-02, 4.957820e-02, 4.010597e-02,
          2.963628e-02, 1.981730e-02, 1.381090e-02, 8.934624e-03, 6.685661e-03],
         [3.600513e-02, 3.804553e-02, 4.107775e-02, 4.680726e-02, 4.518919e-02,
          3.375707e-02, 2.244506e-02, 1.677880e-02, 1.108035e-02, 1.000807e-02]],
        dtype=np.float32)

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

    label_image = iss.label_image

    seg = iss.seg

    # segmentation identifies only one cell
    assert seg._segmentation_instance.num_cells == 1

    # assign targets
    lab = TargetAssignment.Label()
    assigned = lab.run(label_image, decoded)

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
    assert np.sum(assigned['cell_id'] == 1) == 28
