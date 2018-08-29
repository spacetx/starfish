import warnings

import numpy as np

from starfish import Experiment
from starfish.image._filter.white_tophat import WhiteTophat
from starfish.image._registration.fourier_shift import FourierShiftRegistration
from starfish.image._segmentation.watershed import Watershed
from starfish.spots._detector.gaussian import GaussianSpotDetector
from starfish.spots._target_assignment.point_in_poly import PointInPoly2D
from starfish.types import Features, Indices


def test_iss_pipeline_cropped_data():

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    # load the experiment
    experiment_json = (
        "https://dmf0bdeheu4zf.cloudfront.net/20180827/ISS-TEST/experiment.json"
    )
    experiment = Experiment.from_json(experiment_json)
    dots = experiment.fov()['dots']
    nuclei = experiment.fov()['nuclei']
    primary_image = experiment.fov().primary_image

    # register the data
    fsr = FourierShiftRegistration(upsampling=1000, reference_stack=dots)
    registered_primary_image = fsr.run(primary_image, in_place=False)

    # pick a random part of the registered image and assert on it
    expected_registration_values = np.array(
        [[5887, 5846, 5900, 5891, 5857, 6047, 6393, 6435, 6636, 7476],
         [6334, 6197, 6229, 6266, 6195, 6295, 6490, 6575, 6796, 6866],
         [7110, 6704, 6610, 6511, 6408, 6512, 6665, 6718, 7004, 7247],
         [7862, 7172, 7143, 6824, 6746, 6778, 6902, 6905, 6955, 7497],
         [9192, 8233, 7913, 7654, 7501, 7140, 7160, 6966, 7154, 7250],
         [9637, 9302, 9007, 8592, 8018, 7750, 7666, 7339, 7112, 7134],
         [9638, 9787, 9840, 9181, 8638, 8505, 7921, 7454, 7455, 7334],
         [8773, 9603, 9924, 9536, 9047, 8472, 8299, 7704, 7647, 7362],
         [8267, 8829, 9444, 9826, 9362, 8898, 8331, 7970, 7670, 7607],
         [8145, 8300, 8797, 9532, 9877, 9304, 8451, 8141, 7881, 7919]],
        dtype=np.uint16
    )
    assert np.array_equal(
        expected_registration_values,
        registered_primary_image.numpy_array[2, 2, 0, 40:50, 40:50]
    )

    # filter the data
    filt = WhiteTophat(masking_radius=15)
    filtered_dots, filtered_nuclei, filtered_primary_image = [
        filt.run(img, in_place=False) for img in (dots, nuclei, registered_primary_image)
    ]

    # assert on a random part of the filtered image
    expected_filtered_values = np.array(
        [[63, 160, 22, 67, 0, 0, 0, 0, 183, 272],
         [481, 529, 150, 64, 45, 114, 0, 0, 0, 0],
         [1044, 194, 298, 306, 0, 0, 42, 113, 0, 0],
         [2008, 1059, 533, 148, 322, 95, 124, 0, 0, 18],
         [2442, 1557, 1048, 784, 857, 123, 162, 107, 65, 0],
         [2715, 2146, 1739, 1397, 1006, 545, 370, 0, 172, 0],
         [2868, 2771, 2434, 2426, 1641, 1002, 671, 191, 109, 0],
         [2686, 3078, 2984, 2601, 2092, 1439, 882, 531, 218, 0],
         [1992, 2698, 3225, 2832, 2369, 1718, 1478, 1088, 385, 97],
         [1880, 2122, 3027, 3178, 2398, 1652, 1522, 1375, 724, 285]],
        dtype=np.uint16
    )
    assert np.array_equal(
        expected_filtered_values,
        filtered_primary_image.numpy_array[1, 1, 0, 40:50, 40:50]
    )

    # call spots
    gsd = GaussianSpotDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # blobs = dots; define the spots in the dots image, but then find them again in the stack.
        blobs_image = dots.max_proj(Indices.ROUND, Indices.Z)
        intensities = gsd.run(primary_image, blobs_image=blobs_image)

    # assert that the number of spots detected is 99
    assert intensities.sizes[Features.AXIS] == 99

    # decode
    decoded_intensities = experiment.codebook.decode_per_round_max(intensities)

    # decoding identifies 4 genes, each with 1 count
    genes, gene_counts = np.unique(decoded_intensities['target'], return_counts=True)
    assert np.array_equal(genes, np.array(['CD68', 'GUS', 'None', 'ST-3', 'VEGF']))
    assert np.array_equal(gene_counts, [1, 1, 95, 1, 1])

    # segment
    # TODO ambrosejcarr: do these need to be adjusted for the image size?
    seg = Watershed(
        dapi_threshold=.16,
        input_threshold=.22,
        min_distance=57
    )
    regions = seg.run(filtered_primary_image, filtered_nuclei)

    # segmentation identifies only one cell
    assert seg._segmentation_instance.num_cells == 1

    # assign targets
    pip = PointInPoly2D()
    pip.run(decoded_intensities, regions)

    # 18 of the spots are assigned to cell 1 (although most spots do not decode!)
    assert np.sum(decoded_intensities['cell_id'] == 0) == 18
