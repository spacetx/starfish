import os

import pytest

from .util import SpaceTxValidator, package_root

schema_path = os.path.join(package_root, 'schema', 'experiment.json')
validator = SpaceTxValidator(schema_path)
example = os.path.join(package_root, 'examples', 'experiment', 'experiment.json')


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
    dartfish_example = os.path.join(
        package_root, 'examples', 'experiment', 'dartfish_experiment.json')
    assert validator.validate_file(dartfish_example)
