from importlib.resources import files

import pytest
from slicedimage import VERSIONS

from .util import SpaceTxValidator


package_name = "starfish"
current_version = VERSIONS[-1].VERSION

coordinates_path = str(files(package_name).joinpath(
    f"spacetx_format/schema/field_of_view_{current_version}/tiles/coordinates.json"))
validator = SpaceTxValidator(coordinates_path)

example = str(files(package_name).joinpath(
    "spacetx_format/examples/field_of_view/coordinates_%s.json"))


@pytest.mark.parametrize("name", (
    "bad_x_scalar",
    "bad_x_single",
    "bad_y_scalar",
    "bad_y_single",
    "bad_z_single",
    "bad_z_triple",
    "good_max",
    "good_min",
))
def test_coords(name):
    filename = example % name
    if "good" in name:
        assert validator.validate_file(filename)
    else:
        with pytest.warns(UserWarning):
            assert not validator.validate_file(filename)
