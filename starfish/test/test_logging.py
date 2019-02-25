from starfish.util import logging
from starfish.types import CORE_DEPENDENCIES


def test_get_core_dependency_info():
    dependecy_info = logging.get_core_dependency_info()
    for dependecy in CORE_DEPENDENCIES:
        assert dependecy in dependecy_info


def test_get_git_commit_hash():
    print(logging.get_git_commit_hash())


def test_get_os_info():
    print(logging.get_os_info())