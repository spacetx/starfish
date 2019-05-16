import pytest
from pkg_resources import resource_filename

from .util import LatestFOVValidator

package_name = "starfish"
validator = LatestFOVValidator()
example = resource_filename(
    package_name, "spacetx_format/examples/field_of_view/field_of_view.json")
too_large = resource_filename(
    package_name, "spacetx_format/examples/field_of_view/too_large.json")


def test_field_of_view():
    assert validator.validate_file(example)


def test_dartfish_example_field_of_view():
    dartfish_example_path = resource_filename(
        package_name, "spacetx_format/examples/field_of_view/dartfish_field_of_view.json")
    assert validator.validate_file(dartfish_example_path)


def test_dartfish_nuclei_example_field_of_view():
    dartfish_example_path = resource_filename(
        package_name, "spacetx_format/examples/field_of_view/dartfish_nuclei.json")
    assert validator.validate_file(dartfish_example_path)


def test_channel_must_be_present():
    no_channel = validator.load_json(example)
    del no_channel['tiles'][0]['indices']['c']
    with pytest.warns(UserWarning):
        assert not validator.validate_object(no_channel)


def test_round_must_be_present():
    mangled_round = validator.load_json(example)
    del mangled_round['tiles'][0]['indices']['r']
    mangled_round['tiles'][0]['indices']['h'] = 0
    with pytest.warns(UserWarning):
        assert not validator.validate_object(mangled_round)


def test_too_large():
    big = validator.load_json(too_large)
    with pytest.warns(UserWarning):
        assert not validator.validate_object(big)
