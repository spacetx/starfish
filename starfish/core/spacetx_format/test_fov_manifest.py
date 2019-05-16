import pytest
from pkg_resources import resource_filename

from .util import LatestManifestValidator

package_name = "starfish"
validator = LatestManifestValidator()


def test_fov_manifest():
    fov_manifest_example_path = resource_filename(
        package_name, "spacetx_format/examples/fov_manifest/fov_manifest.json")
    assert validator.validate_file(fov_manifest_example_path)


def test_empty_manifest_raises_validation_error():
    with pytest.warns(UserWarning):
        assert not validator.validate_object({})
