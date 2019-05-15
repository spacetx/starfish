import pytest
from pkg_resources import resource_filename

from starfish.core.codebook._format import CURRENT_VERSION, DocumentKeys
from .util import LatestCodebookValidator
from .validate_sptx import validate_file

package_name = "starfish"
validator = LatestCodebookValidator()


def test_codebook():
    codebook_example_path = resource_filename(
        package_name, "spacetx_format/examples/codebook/codebook.json")
    assert validator.validate_file(codebook_example_path)


def test_diagonal_codebook():
    codebook_example_path = resource_filename(
        package_name, "spacetx_format/examples/codebook/codebook_diagonal.json")
    assert validator.validate_file(codebook_example_path)


def test_diagonal_codebook_full_values():
    codebook_example_path = resource_filename(
        package_name, "spacetx_format/examples/codebook/codebook_diagonal_inferred_value.json")
    assert validator.validate_file(codebook_example_path)


def test_codebook_missing_channel_raises_validation_error():
    codebook_example_path = resource_filename(
        package_name, "spacetx_format/examples/codebook/codebook.json")
    codebook = validator.load_json(codebook_example_path)
    del codebook[DocumentKeys.MAPPINGS_KEY][0]['codeword'][0]['c']
    with pytest.warns(UserWarning):
        assert not validator.validate_object(codebook)


def test_codebook_validate():
    example = resource_filename(
        package_name, "spacetx_format/examples/codebook/codebook.json")
    assert validate_file(example, f"codebook_{CURRENT_VERSION}/codebook.json")
