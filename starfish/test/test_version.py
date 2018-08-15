import pytest

from starfish.experiment import Experiment


def test_min_version():
    with pytest.raises(ValueError):
        Experiment.verify_version("0.0.0-dev")
    Experiment.verify_version(str(Experiment.MIN_SUPPORTED_VERSION))
    Experiment.verify_version(str(Experiment.MAX_SUPPORTED_VERSION))


def test_max_version():
    with pytest.raises(ValueError):
        Experiment.verify_version("4294967296.0.0")
    Experiment.verify_version(str(Experiment.MIN_SUPPORTED_VERSION))
    Experiment.verify_version(str(Experiment.MAX_SUPPORTED_VERSION))
