import pytest
from pkg_resources import resource_filename

from .util import SpaceTxValidator

schema_path = resource_filename("validate_sptx", "schema/fov_manifest.json")
validator = SpaceTxValidator(schema_path)


def test_fov_manifest():
    fov_manifest_example_path = resource_filename(
        "validate_sptx", "examples/fov_manifest/fov_manifest.json")
    assert validator.validate_file(fov_manifest_example_path)


def test_empty_manifest_raises_validation_error():
    with pytest.warns(UserWarning):
        assert not validator.validate_object({})
