"""To build experiments in-place, a few things must be done:

1. Call write_experiment_json with writer_contract=InplaceWriterContract().
2. The TileFetcher should return an instance of a InplaceTileFetcher.
"""


import abc
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from slicedimage import ImageFormat, Tile
from slicedimage.io import WriterContract

from starfish.core.types import Axes
from .providers import FetchedTile


class InplaceWriterContract(WriterContract):
    def tile_url_generator(self, tileset_url: str, tile: Tile, ext: str) -> str:
        return tile.provider.filepath.as_uri()

    def write_tile(
            self,
            tile_url: str,
            tile: Tile,
            tile_format: ImageFormat,
            backend_config: Optional[Mapping] = None,
    ) -> str:
        return tile.provider.sha256


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

    def tile_data(self) -> np.ndarray:
        return np.zeros(shape=(self.shape[Axes.Y], self.shape[Axes.X]), dtype=np.float32)
