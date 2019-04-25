import pytest
from pkg_resources import resource_filename

from .util import ManifestValidator

package_name = "starfish"
current_version = "0.1.0"
validator = ManifestValidator(version=current_version)


def test_fov_manifest():
    fov_manifest_example_path = resource_filename(
        package_name, "spacetx_format/examples/fov_manifest/fov_manifest.json")
    assert validator.validate_file(fov_manifest_example_path)


def test_empty_manifest_raises_validation_error():
    with pytest.warns(UserWarning):
        assert not validator.validate_object({})
