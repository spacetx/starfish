import warnings
import tempfile
import os
import numpy as np
import pytest

from starfish.constants import Indices
from starfish.codebook import Codebook
from starfish.intensity_table import IntensityTable

# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    simple_codebook_json, simple_codebook_array, euclidean_decoded_intensities,
    per_channel_max_decoded_intensities, loaded_codebook, small_intensity_table)


def test_loading_codebook_from_json(simple_codebook_json):
    cb = Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=2)
    assert isinstance(cb, Codebook)


def test_loading_codebook_from_list(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array, n_ch=2, n_hyb=2)
    assert isinstance(cb, Codebook)


def test_loading_codebook_without_specifying_ch_hyb_guesses_correct_values(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array)
    assert cb.shape == (3, 2, 2)


def test_loading_codebook_with_too_few_dims_raises_value_error(simple_codebook_json):
    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=1, n_hyb=2)

    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=1)


def test_euclidean_decode_yields_correct_output(euclidean_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = euclidean_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


def test_indexing_on_set_genes(euclidean_decoded_intensities):
    # note that this kind of indexing produces an xarray-internal FutureWarning about float
    # conversion that we can safely ignore here.
    is_actin = euclidean_decoded_intensities[IntensityTable.Constants.GENE.value] == 'ACTB'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        # select only the intensities that are actin, drop the rest
        result = euclidean_decoded_intensities.where(is_actin, drop=True)

    assert result.shape == (1, 2, 2)


def test_synthetic_codes_are_on_only_once_per_channel(euclidean_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = euclidean_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


def test_per_channel_max_decode_yields_expected_results(per_channel_max_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = per_channel_max_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


def test_synthetic_one_hot_codes_produce_one_channel_per_hyb():
    cb = Codebook.synthetic_one_hot_codebook(n_hyb=6, n_channel=4, n_codes=100)
    # sum over channels: only one should be "on"
    assert np.all(cb.sum(Indices.CH.value) == 1)


def test_codebook_save(loaded_codebook):
    directory = tempfile.mkdtemp()
    filename = os.path.join(directory, 'codebook.json')
    loaded_codebook.to_json(filename)
    reloaded = Codebook.from_json(filename, n_hyb=2, n_ch=2)

    assert np.array_equal(loaded_codebook, reloaded)
    assert np.array_equal(loaded_codebook[Indices.CH.value], reloaded[Indices.CH.value])
    assert np.array_equal(loaded_codebook[Indices.HYB.value], reloaded[Indices.HYB.value])
    assert np.array_equal(loaded_codebook[Codebook.Constants.GENE.value].values,
                          reloaded[Codebook.Constants.GENE.value].values)


@pytest.mark.parametrize('n_ch, n_hyb', ((2, 2), (5, 4)))
def test_loading_codebook_with_unused_channels_and_hybs(simple_codebook_json, n_ch, n_hyb):
    cb = Codebook.from_json(simple_codebook_json, n_ch=n_ch, n_hyb=n_hyb)
    assert cb.shape == (3, n_ch, n_hyb)


@pytest.mark.parametrize('n_ch, n_hyb', ((2, 2), (5, 4)))
def test_code_length(n_ch, n_hyb):
    gene_names = np.arange(10)
    cb = Codebook._empty_codebook(gene_names, n_ch, n_hyb)
    assert cb.code_length == n_ch * n_hyb
