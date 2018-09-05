import numpy as np
import pandas as pd

from starfish import Codebook, Experiment
from starfish.image._filter.scale_by_percentile import ScaleByPercentile
from starfish.image._filter.zero_by_channel_magnitude import ZeroByChannelMagnitude
from starfish.spots._detector.pixel_spot_detector import PixelSpotDetector
from starfish.types import Features


def test_dartfish_pipeline_cropped_data():

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    # load the experiment
    experiment_json = (
        "https://dmf0bdeheu4zf.cloudfront.net/20180828/DARTFISH-TEST/experiment.json"
    )
    experiment = Experiment.from_json(experiment_json)

    primary_image = experiment.fov().primary_image
    primary_image._data = primary_image._data.astype(float)

    expected_primary_image = np.array(
        [[1., 0., 0., 0., 0., 0., 3., 0., 0., 2.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 2., 0.],
         [0., 0., 1., 0., 0., 1., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 2., 0., 1., 0.],
         [0., 0., 1., 0., 2., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.]],
        dtype=np.float
    )

    assert np.array_equal(
        primary_image.numpy_array[0, 0, 0, 50:60, 60:70],
        expected_primary_image
    )

    sc_filt = ScaleByPercentile(p=100)
    normalized_image = sc_filt.run(primary_image, in_place=False)

    expected_normalized_image = np.array(
        [[0.01960784, 0., 0., 0., 0.,
          0., 0.05882353, 0., 0., 0.03921569],
         [0., 0., 0., 0., 0.01960784,
          0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.01960784,
          0., 0., 0., 0.03921569, 0.],
         [0., 0., 0.01960784, 0., 0.,
          0.01960784, 0., 0.01960784, 0., 0.],
         [0., 0., 0., 0., 0.,
          0., 0.03921569, 0., 0.01960784, 0.],
         [0., 0., 0.01960784, 0., 0.03921569,
          0., 0., 0., 0., 0.01960784],
         [0., 0., 0., 0., 0.,
          0.01960784, 0., 0., 0., 0.],
         [0., 0.01960784, 0., 0., 0.01960784,
          0., 0., 0., 0., 0.],
         [0.01960784, 0.01960784, 0., 0., 0.,
          0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.,
          0.01960784, 0.01960784, 0., 0., 0.]],
        dtype=np.float,
    )

    assert np.allclose(
        normalized_image.numpy_array[0, 0, 0, 50:60, 60:70],
        expected_normalized_image
    )

    z_filt = ZeroByChannelMagnitude(thresh=.05, normalize=False)
    zero_norm_stack = z_filt.run(normalized_image, in_place=False)

    expected_zero_normalized_image = np.array(
        [[0.01960784, 0., 0., 0., 0.,
          0., 0.05882353, 0., 0., 0.03921569],
         [0., 0., 0., 0., 0.01960784,
          0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.01960784,
          0., 0., 0., 0., 0.],
         [0., 0., 0.01960784, 0., 0.,
          0., 0., 0.01960784, 0., 0.],
         [0., 0., 0., 0., 0.,
          0., 0.03921569, 0., 0.01960784, 0.],
         [0., 0., 0.01960784, 0., 0.,
          0., 0., 0., 0., 0.01960784],
         [0., 0., 0., 0., 0.,
          0.01960784, 0., 0., 0., 0.],
         [0., 0.01960784, 0., 0., 0.,
          0., 0., 0., 0., 0.],
         [0.01960784, 0., 0., 0., 0.,
          0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.,
          0., 0.01960784, 0., 0., 0.]],
        dtype=np.float
    )

    assert np.allclose(
        expected_zero_normalized_image,
        zero_norm_stack.numpy_array[0, 0, 0, 50:60, 60:70]
    )

    magnitude_threshold = 0.5
    area_threshold = (5, 30)
    distance_threshold = 3

    psd = PixelSpotDetector(
        codebook=experiment.codebook,
        metric='euclidean',
        distance_threshold=distance_threshold,
        magnitude_threshold=magnitude_threshold,
        min_area=area_threshold[0],
        max_area=area_threshold[1]
    )

    spot_intensities, results = psd.run(zero_norm_stack)
    spots_df = spot_intensities.to_features_dataframe()
    spots_df['area'] = np.pi * spots_df['radius'] ** 2

    # verify number of spots detected
    assert spot_intensities.sizes[Features.AXIS] == 53

    # compare to benchmark data -- note that this particular part of the dataset appears completely
    # uncorrelated
    cnts_benchmark = pd.read_csv(
        'https://dmf0bdeheu4zf.cloudfront.net/20180813/DARTFISH/fov_001/counts.csv')

    min_dist = 0.6
    cnts_starfish = spots_df[spots_df.distance <= min_dist].groupby('target').count()['area']
    cnts_starfish = cnts_starfish.reset_index(level=0)
    cnts_starfish.rename(columns={'target': 'gene', 'area': 'cnt_starfish'}, inplace=True)

    # get top 5 genes and verify they are correct
    high_expression_genes = cnts_starfish.sort_values('cnt_starfish', ascending=False).head(5)
    assert np.array_equal(
        high_expression_genes['cnt_starfish'].values,
        [7, 3, 2, 2, 2]
    )
    assert np.array_equal(
        high_expression_genes['gene'].values,
        ['MBP', 'MOBP', 'ADCY8', 'TRIM66', 'SYT6']
    )

    # verify correlation is accurate for this subset of the image
    benchmark_comparison = pd.merge(cnts_benchmark, cnts_starfish, on='gene', how='left')
    benchmark_comparison.head(20)

    x = benchmark_comparison.dropna().cnt.values
    y = benchmark_comparison.dropna().cnt_starfish.values
    corrcoef = np.corrcoef(x, y)
    corrcoef = corrcoef[0, 1]

    assert np.round(corrcoef, 5) == -0.00992
