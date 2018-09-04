import os

import pytest

from .util import SpaceTxValidator, package_root

schema_path = os.path.join(package_root, 'schema', 'fov_manifest.json')
validator = SpaceTxValidator(schema_path)


def test_fov_manifest():
    example = os.path.join(
        package_root, 'examples', 'fov_manifest', 'fov_manifest.json'
    )
    assert validator.validate_file(example)


def test_empty_manifest_raises_validation_error():
    with pytest.warns(UserWarning):
        assert not validator.validate_object({})
