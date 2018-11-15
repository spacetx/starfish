import pytest
from pkg_resources import resource_filename

from .util import SpaceTxValidator

package_name = "sptx_format"
coordinates_path = resource_filename(package_name, "schema/field_of_view/tiles/coordinates.json")
validator = SpaceTxValidator(coordinates_path)

example = resource_filename(package_name, "examples/field_of_view/coordinates%s.json")

@pytest.mark.parametrize("args", (
    ("1", True),
    ("2", True),
    ("3", True),
    ("4", False),
))
def test_coords(args):
    index, passes = args
    filename = example % index
    if passes:
        assert validator.validate_file(filename)
    else:
        with pytest.warns(UserWarning):
            assert not validator.validate_file(filename)
