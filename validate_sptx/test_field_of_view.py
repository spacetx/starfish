import pytest
from pkg_resources import resource_filename

from .util import SpaceTxValidator

fov_schema_path = resource_filename("validate_sptx", "schema/field_of_view/field_of_view.json")
validator = SpaceTxValidator(fov_schema_path)
example = resource_filename("validate_sptx", "examples/field_of_view/field_of_view.json")


def test_field_of_view():
    assert validator.validate_file(example)


def test_dartfish_example_field_of_view():
    dartfish_example_path = resource_filename(
        "validate_sptx", "examples/field_of_view/dartfish_field_of_view.json")
    assert validator.validate_file(dartfish_example_path)


def test_dartfish_nuclei_example_field_of_view():
    dartfish_example_path = resource_filename(
        "validate_sptx", "examples/field_of_view/dartfish_nuclei.json")
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
