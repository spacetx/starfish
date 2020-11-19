import os
import sys
import tempfile

import numpy as np
import pandas as pd

import starfish
from starfish import IntensityTable
from starfish.types import Coordinates, Features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(starfish.__file__)))
os.environ["USE_TEST_DATA"] = "1"
sys.path.append(os.path.join(ROOT_DIR, "notebooks", "py"))


def test_dartfish_pipeline_cropped_data(tmpdir):

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    dartfish = __import__('DARTFISH')

    primary_image = dartfish.imgs

    expected_primary_image = np.array(
        [[1.52590219e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 4.57770657e-05, 0.00000000e+00,
          0.00000000e+00, 3.05180438e-05],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.52590219e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.52590219e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          3.05180438e-05, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 1.52590219e-05, 0.00000000e+00,
          0.00000000e+00, 1.52590219e-05, 0.00000000e+00, 1.52590219e-05,
          0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 3.05180438e-05, 0.00000000e+00,
          1.52590219e-05, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 1.52590219e-05, 0.00000000e+00,
          3.05180438e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 1.52590219e-05],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 1.52590219e-05, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 1.52590219e-05, 0.00000000e+00, 0.00000000e+00,
          1.52590219e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00],
         [1.52590219e-05, 1.52590219e-05, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 1.52590219e-05, 1.52590219e-05, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00]],
        dtype=np.float32
    )

    assert primary_image.xarray.dtype == np.float32

    assert np.allclose(
        primary_image.xarray[0, 0, 0, 50:60, 60:70],
        expected_primary_image
    )

    normalized_image = dartfish.norm_imgs

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
        dtype=np.float32,
    )
    assert normalized_image.xarray.dtype == np.float32

    assert np.allclose(
        normalized_image.xarray[0, 0, 0, 50:60, 60:70],
        expected_normalized_image
    )

    zero_norm_stack = dartfish.filtered_imgs

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
        dtype=np.float32
    )

    assert np.allclose(
        expected_zero_normalized_image,
        zero_norm_stack.xarray[0, 0, 0, 50:60, 60:70]
    )

    pipeline_log = zero_norm_stack.log.data

    assert pipeline_log[0]['method'] == 'Clip'
    assert pipeline_log[1]['method'] == 'ZeroByChannelMagnitude'

    spot_intensities = dartfish.initial_spot_intensities

    # assert tht physical coordinates were transferred
    assert Coordinates.X in spot_intensities.coords
    assert Coordinates.Y in spot_intensities.coords
    assert Coordinates.Z in spot_intensities.coords

    pipeline_log = spot_intensities.get_log()

    assert pipeline_log[0]['method'] == 'Clip'
    assert pipeline_log[1]['method'] == 'ZeroByChannelMagnitude'

    # Test serialization / deserialization of IntensityTable log

    with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False) as ntf:
        tfp = ntf.name
    spot_intensities.to_netcdf(tfp)

    loaded_intensities = IntensityTable.open_netcdf(tfp)
    pipeline_log = loaded_intensities.get_log()

    assert pipeline_log[0]['method'] == 'Clip'
    assert pipeline_log[1]['method'] == 'ZeroByChannelMagnitude'

    spots_df = IntensityTable(
        spot_intensities.where(spot_intensities[Features.PASSES_THRESHOLDS], drop=True)
    ).to_features_dataframe()
    spots_df['area'] = np.pi * spots_df['radius'] ** 2

    # verify number of spots detected
    spots_passing_filters = spot_intensities[Features.PASSES_THRESHOLDS].sum()
    assert spots_passing_filters == 53

    # compare to benchmark data -- note that this particular part of the dataset appears completely
    # uncorrelated
    # instead of just calling read_csv with the url, we are using python requests to load it to
    # avoid a SSL certificate error on some platforms.
    import io, requests
    cnts_benchmark = pd.read_csv(io.BytesIO(requests.get(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH/fov_001/counts.csv').content))

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

    assert np.round(corrcoef, 5) == 0.03028
