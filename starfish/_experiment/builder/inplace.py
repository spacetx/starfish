"""To build experiments in-place, a few things must be done:

1. Call enable_inplace_mode().
2. Call write_experiment_json with tile_opener=inplace_tile_opener.
3. The TileFetcher should return an instance of a InplaceTileFetcher.

Please note that enabling in-place experiment construction should be only done in an isolated script
dedicated to constructing an experiment, as it modifies some existing code paths.
"""


import abc
import io
import sys
from pathlib import Path
from typing import BinaryIO

from slicedimage import Tile

from .providers import FetchedTile


def sha256_get(tile_self):
    return tile_self.provider.sha256


def sha256_set(tile_self, value):
    pass


def enable_inplace_mode():
    Tile.sha256 = property(sha256_get, sha256_set)


def inplace_tile_opener(toc_path: Path, tile: Tile, file_ext: str) -> BinaryIO:
    return DevNull(tile.provider.filepath)


class DevNull(io.BytesIO):
    """A class meant to mimic an open(filepath, 'wb') operation but
    prevents any actual writing, reading, or seeking.
    This class is an ugly hack to prevent the slicedimage
    Writer.generate_partition_document() function from needlessly creating
    a copy of image data.
    See: https://docs.python.org/3/library/io.html
    See also: cpython/Lib/_pyio.py
    """
    def __init__(self, filepath: str, *args, **kwargs):
        super(DevNull, self).__init__(*args, **kwargs)
        self.name = filepath

    def read(self, size=-1):
        raise NotImplementedError()

    def write(self, b):
        return sys.getsizeof(b)

    def seek(self, pos, whence=0):
        raise NotImplementedError()


class InplaceFetchedTile(FetchedTile):
    """Data formatters that operate in-place should return tiles that extend this class."""
    @property
    @abc.abstractmethod
    def filepath(self) -> Path:
        """Returns the path of the source tile."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def sha256(self):
        """Returns the sha256 checksum of the source tile."""
        raise NotImplementedError()
