import platform
from functools import lru_cache
from json import JSONEncoder
from subprocess import (
    check_output,
)

import pkg_resources

from starfish.types import CORE_DEPENDENCIES


@lru_cache(maxsize=1)
def get_core_dependency_info():
    dependency_info = dict()
    for dependency in CORE_DEPENDENCIES:
        version = pkg_resources.get_distribution(dependency).version
        dependency_info[dependency] = version
    return dependency_info


@lru_cache(maxsize=1)
def get_git_commit_hash():
    # First check for git repo
    return check_output(["git", "describe", "--always"]).strip()


@lru_cache(maxsize=1)
def get_os_info():
    return {"Platform": platform.system(),
            "Version:": platform.version(),
            "Python Version": platform.python_version()}


class LogEncoder(JSONEncoder):
    """
    JSON encodes the List[Dict] pipeline provence log. For simple
    objects use default JSON encoding. For more complex objects
    (ex. Imagestack, Codebook) encode the repr of the object.
    """
    def default(self, o):
        try:
            return super(LogEncoder, self).default(o)
        except TypeError:
            return JSONEncoder().encode(repr(o))
