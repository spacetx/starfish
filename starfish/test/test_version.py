import pytest

from starfish.io import Stack


def test_min_version():
    with pytest.raises(ValueError):
        Stack.verify_version("0.0.0-dev")
    Stack.verify_version(str(Stack.MIN_SUPPORTED_VERSION))
    Stack.verify_version(str(Stack.MAX_SUPPORTED_VERSION))


def test_max_version():
    with pytest.raises(ValueError):
        Stack.verify_version("4294967296.0.0")
    Stack.verify_version(str(Stack.MIN_SUPPORTED_VERSION))
    Stack.verify_version(str(Stack.MAX_SUPPORTED_VERSION))
