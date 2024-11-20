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
        [[8.182720e-03, 2.094890e-03, 1.921310e-03, 8.036800e-04, 0.000000e+00,
          1.113000e-05, 2.362500e-04, 1.137100e-04, 2.881270e-03, 1.061417e-02],
         [1.225884e-02, 7.359780e-03, 6.554780e-03, 3.967060e-03, 7.174400e-04,
          1.377300e-04, 3.282700e-04, 1.963200e-04, 1.156900e-04, 2.786610e-03],
         [1.817651e-02, 1.313646e-02, 1.080875e-02, 8.000580e-03, 2.264170e-03,
          8.996000e-04, 1.071100e-03, 2.864700e-04, 4.592600e-04, 2.591370e-03],
         [2.938030e-02, 2.027904e-02, 1.634731e-02, 1.067738e-02, 5.404920e-03,
          3.325540e-03, 2.005550e-03, 3.105000e-05, 9.240000e-04, 5.751660e-03],
         [4.422882e-02, 3.153970e-02, 2.650198e-02, 1.768988e-02, 1.198124e-02,
          7.287380e-03, 4.359240e-03, 4.564000e-05, 1.156120e-03, 3.979250e-03],
         [5.676676e-02, 4.604779e-02, 3.936250e-02, 2.943260e-02, 1.997995e-02,
          1.306023e-02, 9.153120e-03, 1.940280e-03, 1.672590e-03, 1.607550e-03],
         [5.948846e-02, 5.680262e-02, 5.028814e-02, 3.747543e-02, 2.800228e-02,
          2.101545e-02, 1.274632e-02, 5.316650e-03, 4.062480e-03, 1.875220e-03],
         [5.272433e-02, 5.822361e-02, 5.484088e-02, 4.344586e-02, 3.279146e-02,
          2.484935e-02, 1.552278e-02, 8.065560e-03, 8.012830e-03, 2.026330e-03],
         [4.264575e-02, 5.009904e-02, 5.163100e-02, 4.824880e-02, 3.619050e-02,
          2.628749e-02, 1.851076e-02, 1.187091e-02, 8.305970e-03, 4.661620e-03],
         [3.705096e-02, 4.012012e-02, 4.393976e-02, 4.851252e-02, 4.276346e-02,
          3.062118e-02, 1.944861e-02, 1.515226e-02, 9.333680e-03, 9.347120e-03]],
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

    assert np.array_equal(gene_counts, [19, 1, 5, 2, 1, 9, 1, 3, 2, 1, 1, 2])
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
    assert np.array_equal(expression_matrix.number_of_undecoded_spots.data, [14, 1, 37])
