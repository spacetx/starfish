import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from starfish.constants import Indices
from starfish.image import ImageStack
from starfish.io import Stack
from starfish.munge import dataframe_to_multiindex
from starfish.codebook import Codebook
from starfish.intensity_table import IntensityTable
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.util import synthesize


# TODO ambrosejcarr: all fixtures should emit a stack and a codebook
@pytest.fixture(scope='session')
def merfish_stack() -> Stack:
    """retrieve MERFISH testing data from cloudfront and expose it at the module level

    Notes
    -----
    Because download takes time, this fixture runs once per session -- that is, the download is run only once.

    Returns
    -------
    Stack :
        starfish.io.Stack object containing MERFISH data
    """
    s = Stack()
    s.read('https://s3.amazonaws.com/czi.starfish.data.public/20180607/test/MERFISH/fov_001/experiment_new.json')
    return deepcopy(s)


@pytest.fixture(scope='session')
def labeled_synthetic_dataset():
    stp = synthesize.SyntheticSpotTileProvider()
    image = ImageStack.synthetic_stack(tile_data_provider=stp.tile)
    max_proj = image.max_proj(Indices.HYB, Indices.CH, Indices.Z)
    view = max_proj.reshape((1, 1, 1) + max_proj.shape)
    dots = ImageStack.from_numpy_array(view)

    def labeled_synthetic_dataset_factory():
        return deepcopy(image), deepcopy(dots), deepcopy(stp.codebook)

    return labeled_synthetic_dataset_factory


@pytest.fixture(scope='function')
def small_intensity_table():
    intensities = np.array([
        [[0, 1],
         [1, 0]],
        [[1, 0],
         [0, 1]],
        [[0, 0],
         [1, 1]]
    ])

    spot_attributes = dataframe_to_multiindex(pd.DataFrame(
        data={
            IntensityTable.SpotAttributes.X: [0, 1, 2],
            IntensityTable.SpotAttributes.Y: [3, 4, 5],
            IntensityTable.SpotAttributes.Z: [0, 0, 0],
            IntensityTable.SpotAttributes.RADIUS: [0.1, 0.2, 0.3]
        }
    ))

    return IntensityTable.from_spot_data(intensities, spot_attributes)


@pytest.fixture(scope='module')
def simple_codebook_array():
    return [
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 0, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "SCUBE2"
        },
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "BRCA"
        },
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 0, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "ACTB"
        }
    ]


@pytest.fixture(scope='module')
def simple_codebook_json(simple_codebook_array) -> Generator[str, None, None]:
    directory = tempfile.mkdtemp()
    codebook_json = os.path.join(directory, 'simple_codebook.json')
    with open(codebook_json, 'w') as f:
        json.dump(simple_codebook_array, f)

    yield codebook_json

    shutil.rmtree(directory)


@pytest.fixture(scope='module')
def loaded_codebook(simple_codebook_json):
    return Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=2)


@pytest.fixture(scope='function')
def euclidean_decoded_intensities(small_intensity_table, loaded_codebook):
    decoded_intensities = loaded_codebook.decode_euclidean(small_intensity_table)
    return decoded_intensities


@pytest.fixture(scope='function')
def per_channel_max_decoded_intensities(small_intensity_table, loaded_codebook):
    decoded_intensities = loaded_codebook.decode_per_hyb_max(small_intensity_table)
    return decoded_intensities


@pytest.fixture(scope='module')
def synthetic_intensity_table(loaded_codebook) -> IntensityTable:
    return IntensityTable.synthetic_intensities(loaded_codebook, n_spots=2)


@pytest.fixture(scope='module')
def synthetic_dataset_with_truth_values():
    from starfish.util.synthesize import SyntheticData

    np.random.seed(2)
    synthesizer = SyntheticData(n_spots=5)
    codebook = synthesizer.codebook()
    true_intensities = synthesizer.intensities(codebook=codebook)
    image = synthesizer.spots(intensities=true_intensities)

    return codebook, true_intensities, image


@pytest.fixture(scope='function')
def synthetic_dataset_with_truth_values_and_called_spots(
        synthetic_dataset_with_truth_values
):

    codebook, true_intensities, image = synthetic_dataset_with_truth_values

    wth = WhiteTophat(disk_size=15)
    filtered = wth.filter(image, in_place=False)

    min_sigma = 1.5
    max_sigma = 4
    num_sigma = 10
    threshold = 1e-4
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=filtered,
        measurement_type='max',
    )

    intensities = gsd.find(hybridization_image=filtered)
    assert intensities.shape[0] == 5

    codebook.decode_euclidean(intensities)

    return codebook, true_intensities, image, intensities


@pytest.fixture()
def synthetic_single_spot_2d():
    from scipy.ndimage.filters import gaussian_filter
    data = np.zeros((100, 100), dtype=np.uint16)
    data[10, 90] = 1000
    data = gaussian_filter(data, sigma=2)
    return data


@pytest.fixture()
def synthetic_single_spot_3d():
    from scipy.ndimage.filters import gaussian_filter
    data = np.zeros((10, 100, 100), dtype=np.uint16)
    data[5, 10, 90] = 1000
    data = gaussian_filter(data, sigma=2)
    return data


@pytest.fixture()
def synthetic_two_spot_3d():
    from scipy.ndimage.filters import gaussian_filter
    data = np.zeros((10, 100, 100), dtype=np.uint16)
    data[4, 10, 90] = 1000
    data[6, 90, 10] = 1000
    data = gaussian_filter(data, sigma=2)
    return data


@pytest.fixture()
def synthetic_single_spot_imagestack_2d(synthetic_single_spot_2d):
    data = synthetic_single_spot_2d
    return ImageStack.from_numpy_array(data.reshape(1, 1, 1, *data.shape))


@pytest.fixture()
def synthetic_single_spot_imagestack_3d(synthetic_single_spot_3d):
    data = synthetic_single_spot_3d
    return ImageStack.from_numpy_array(data.reshape(1, 1, *data.shape))


@pytest.fixture()
def synthetic_two_spot_imagestack_3d(synthetic_two_spot_3d):
    data = synthetic_two_spot_3d
    return ImageStack.from_numpy_array(data.reshape(1, 1, *data.shape))


@pytest.fixture()
def synthetic_spot_pass_through_stack(synthetic_dataset_with_truth_values):
    codebook, true_intensities, _ = synthetic_dataset_with_truth_values
    true_intensities = true_intensities[:2]
    # transfer the intensities to the stack but don't do anything to them.
    img_stack = ImageStack.synthetic_spots(
        true_intensities, num_z=12, height=50, width=45, n_photons_background=0,
        point_spread_function=(0, 0, 0), camera_detection_efficiency=1.0,
        background_electrons=0, graylevel=1)
    return codebook, true_intensities, img_stack
