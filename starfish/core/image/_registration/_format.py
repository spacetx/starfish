"""Constants to support formatting for serialization and deserialization of transforms list."""

from semantic_version import Version

CURRENT_VERSION = Version("0.0.0")
MIN_SUPPORTED_VERSION = Version("0.0.0")
MAX_SUPPORTED_VERSION = Version("0.0.0")


class DocumentKeys:
    VERSION_KEY = 'version'
    TRANSFORMS_LIST = 'transforms_list'
