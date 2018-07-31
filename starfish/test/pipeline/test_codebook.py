import warnings
import tempfile
import json
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import xarray as xr

from starfish.constants import Indices, Features
from starfish.codebook import Codebook

# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    simple_codebook_json, simple_codebook_array, euclidean_decoded_intensities,
    per_channel_max_decoded_intensities, loaded_codebook, small_intensity_table)


def test_loading_codebook_from_json_local_file(simple_codebook_json):
    cb = Codebook.from_json(simple_codebook_json, n_ch=2, n_round=2)
    assert isinstance(cb, Codebook)


@patch('starfish.codebook.urllib.request.urlopen')
def test_loading_codebook_from_json_https_file(mock_urlopen):

    # codebook data to pass to the mock
    _return_value = json.dumps(
        [
            {
                Features.CODEWORD: [
                    {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                    {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
                ],
                Features.TARGET: "SCUBE2"
            }
        ]
    ).encode()

    # mock urlopen.read() to return data corresponding to a codebook
    a = MagicMock()
    a.read.side_effect = [_return_value]
    a.__enter__.return_value = a
    mock_urlopen.return_value = a

    # test that the function loads the codebook from the link when called
    cb = Codebook.from_json('https://www.alink.com/file.json', n_ch=2, n_round=2)
    assert isinstance(cb, Codebook)
    assert mock_urlopen.call_count == 1


def test_loading_codebook_from_list(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array, n_ch=2, n_round=2)
    assert isinstance(cb, Codebook)


def test_loading_codebook_without_specifying_ch_round_guesses_correct_values(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array)
    assert cb.shape == (3, 2, 2)


def test_loading_codebook_with_too_few_dims_raises_value_error(simple_codebook_json):
    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=1, n_round=2)

    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=2, n_round=1)


def test_euclidean_decode_yields_correct_output(euclidean_decoded_intensities):
    expected_target_annotation = np.array(["ACTB", "SCUBE2", "BRCA", "None", "SCUBE2"])
    observed_target_annotation = euclidean_decoded_intensities[
        Features.TARGET].values
    expected_distances = np.array([0, 0, 0, 0.76536686, 0])
    observed_distances = euclidean_decoded_intensities[
        Features.DISTANCE].values
    assert np.array_equal(expected_target_annotation, observed_target_annotation)
    assert np.allclose(expected_distances, observed_distances)


def test_indexing_on_set_targets(euclidean_decoded_intensities):
    # note that this kind of indexing produces an xarray-internal FutureWarning about float
    # conversion that we can safely ignore here.
    is_actin = euclidean_decoded_intensities[Features.TARGET] == 'ACTB'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        # select only the intensities that are actin, drop the rest
        result = euclidean_decoded_intensities.where(is_actin, drop=True)

    assert result.shape == (1, 2, 2)


def test_per_channel_max_decode_yields_expected_results(per_channel_max_decoded_intensities):
    expected_target_annotation = np.array(["ACTB", "SCUBE2", "BRCA", "None", "SCUBE2"])
    observed_target_annotation = per_channel_max_decoded_intensities[
        Features.TARGET].values
    assert np.array_equal(expected_target_annotation, observed_target_annotation)


def test_synthetic_one_hot_codes_produce_one_channel_per_round():
    cb = Codebook.synthetic_one_hot_codebook(n_round=6, n_channel=4, n_codes=100)
    # sum over channels: only one should be "on"
    assert np.all(cb.sum(Indices.CH.value) == 1)


def test_codebook_save(loaded_codebook):
    directory = tempfile.mkdtemp()
    filename = os.path.join(directory, 'codebook.json')
    loaded_codebook.to_json(filename)
    reloaded = Codebook.from_json(filename, n_round=2, n_ch=2)

    assert np.array_equal(loaded_codebook, reloaded)
    assert np.array_equal(loaded_codebook[Indices.CH.value], reloaded[Indices.CH.value])
    assert np.array_equal(loaded_codebook[Indices.ROUND.value], reloaded[Indices.ROUND.value])
    assert np.array_equal(loaded_codebook[Features.TARGET].values,
                          reloaded[Features.TARGET].values)


@pytest.mark.parametrize('n_ch, n_round', ((2, 2), (5, 4)))
def test_loading_codebook_with_unused_channels_and_rounds(simple_codebook_json, n_ch, n_round):
    cb = Codebook.from_json(simple_codebook_json, n_ch=n_ch, n_round=n_round)
    assert cb.shape == (3, n_ch, n_round)


@pytest.mark.parametrize('n_ch, n_round', ((2, 2), (5, 4)))
def test_code_length(n_ch, n_round):
    target_names = np.arange(10)
    cb = Codebook._empty_codebook(target_names, n_ch, n_round)
    assert cb.code_length == n_ch * n_round


def test_unit_normalize():
    simple_data = np.array(
        [[[0, 1],
          [2, 1]],
         [[0, 0],
          [0, 0]],
         [[9, 0],
          [0, 1]]]
    )
    simple_codebook = Codebook(
        data=simple_data,
        dims=([Features.AXIS, Indices.CH.value, Indices.ROUND.value]),
        coords=(list('xyz'), list('vw'), list('tu'))
    )

    results, _ = Codebook._normalize_features(simple_codebook, norm_order=1)
    assert np.array_equal(results.sum([Indices.CH, Indices.ROUND]), np.ones(3))
