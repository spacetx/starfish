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
        "https://dmf0bdeheu4zf.cloudfront.net/20180919/ISS-TEST/experiment.json"
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
        [[0.08983251, 0.0892154, 0.09003078, 0.08990231, 0.08938038,
          0.09227149, 0.09755211, 0.09819349, 0.10127072, 0.11408974],
         [0.09666385, 0.09456035, 0.09506275, 0.09561637, 0.09453626,
          0.09606067, 0.09903451, 0.1003365, 0.10371296, 0.10477841],
         [0.10850635, 0.1023093, 0.10086684, 0.09936101, 0.09779496,
          0.09937705, 0.10170354, 0.1025213, 0.10687973, 0.11058433],
         [0.1199711, 0.10944109, 0.10901033, 0.10414005, 0.10293939,
          0.10342702, 0.10531835, 0.10537095, 0.10613398, 0.11440884],
         [0.14026444, 0.1256294, 0.12075542, 0.11679934, 0.11446859,
          0.10895271, 0.10925814, 0.10629877, 0.1091761, 0.11063146],
         [0.14706053, 0.14194803, 0.1374521, 0.13111584, 0.12235309,
          0.11825943, 0.11697862, 0.11199736, 0.10853392, 0.10885843],
         [0.14707456, 0.14934249, 0.15015754, 0.14010485, 0.1318078,
          0.12978303, 0.1208757, 0.11374975, 0.11375835, 0.11191528],
         [0.13387902, 0.14654579, 0.15143262, 0.1455235, 0.13806314,
          0.12928307, 0.12664636, 0.11756866, 0.11668609, 0.11233816],
         [0.12615235, 0.13472537, 0.14411529, 0.14993921, 0.14286724,
          0.13577826, 0.127131, 0.1216145, 0.11704073, 0.11607594],
         [0.12429121, 0.12665416, 0.13423485, 0.14545396, 0.15071892,
          0.1419759, 0.1289691, 0.1242372, 0.12026623, 0.12084184]],
        dtype=np.float
    )
    assert np.allclose(
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
        [[0.00095978, 0.00242981, 0.00034942, 0.00102001, 0.,
          0., 0., 0., 0.00279868, 0.00414499],
         [0.00732663, 0.00806766, 0.00228394, 0.00097898, 0.00068559,
          0.00174595, 0., 0., 0., 0.],
         [0.01592878, 0.00296821, 0.00454778, 0.00466756, 0.,
          0., 0.00064306, 0.0017258, 0., 0.],
         [0.030638, 0.01616477, 0.00813692, 0.00225843, 0.00491835,
          0.0014467, 0.0018941, 0., 0., 0.00026727],
         [0.037273, 0.02376178, 0.0159938, 0.01196495, 0.0130818,
          0.00187735, 0.00246782, 0.00162887, 0.0010042, 0.],
         [0.0414216, 0.03275366, 0.02653611, 0.02131327, 0.0153529,
          0.0083164, 0.00564933, 0., 0.00263396, 0.],
         [0.04375885, 0.04228986, 0.03714352, 0.03701786, 0.02504258,
          0.01528613, 0.01023919, 0.00292638, 0.00166556, 0.],
         [0.04098595, 0.0469675, 0.04553796, 0.03968509, 0.03191137,
          0.02195033, 0.01346012, 0.00810439, 0.00333658, 0.],
         [0.03039079, 0.04116304, 0.04920747, 0.04320574, 0.03614666,
          0.02622336, 0.02255489, 0.01660567, 0.00587476, 0.00147908],
         [0.02868421, 0.03238469, 0.04618791, 0.0484842, 0.03658872,
          0.02521646, 0.0232316, 0.02098654, 0.01103822, 0.00434919]],
        dtype=np.float
    )
    assert np.allclose(
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
    assert np.array_equal(genes, np.array(['CD68', 'GUS', 'ST-3', 'VEGF', 'nan']))
    assert np.array_equal(gene_counts, [1, 1, 1, 1, 95])

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
