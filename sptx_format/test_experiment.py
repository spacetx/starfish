import pytest
from pkg_resources import resource_filename

from .util import SpaceTxValidator
package_name = "sptx_format"
experiment_schema_path = resource_filename(package_name, "schema/experiment.json")
validator = SpaceTxValidator(experiment_schema_path)
example = resource_filename(package_name, "examples/experiment/experiment.json")


def test_fov():
    assert validator.validate_file(example)


def test_nuclei_must_be_present():
    wrong_nuclei = validator.load_json(example)
    wrong_nuclei['auxiliary_images'] = {'not_nuclei': 'nuclei.json'}
    with pytest.warns(UserWarning):
        assert not validator.validate_object(wrong_nuclei)


def test_version_must_be_semantic():
    wrong_version = validator.load_json(example)
    wrong_version['version'] = '10a'
    with pytest.warns(UserWarning):
        assert not validator.validate_object(wrong_version)


def test_dartfish_example_experiment():
    dartfish_example = resource_filename(
        package_name, "examples/experiment/dartfish_experiment.json")
    assert validator.validate_file(dartfish_example)
