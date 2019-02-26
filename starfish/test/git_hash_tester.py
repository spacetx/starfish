import sys
from starfish.util import logging


def test_get_git_commit_hash(install_type):
    print(install_type)
    print(logging.get_git_commit_hash())


if __name__ == "__main__":
    test_get_git_commit_hash(sys.argv[1])
