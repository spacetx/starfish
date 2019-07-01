"""
This module implements default providers of data to the experiment builders.
"""

from typing import Mapping, Type, Union

import numpy as np
from slicedimage import (
    ImageFormat,
)

from starfish.core.types import Axes, Coordinates, CoordinateValue
from .providers import FetchedTile, TileFetcher


class RandomNoiseTile(FetchedTile):
    """
    This is a simple implementation of :class:`.FetchedImage` that simply regenerates random data
    for the image.
    """
    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 1536, Axes.X: 1024}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.random.randint(
            0, 256, size=(self.shape[Axes.Y], self.shape[Axes.X]), dtype=np.uint8)


class OnesTile(FetchedTile):
    """
    This is a simple implementation of :class:`.FetchedImage` that simply is entirely all pixels at
    maximum intensity.
    """
    def __init__(self, shape: Mapping[Axes, int]) -> None:
        super().__init__()
        self._shape = shape

    @property
    def shape(self) -> Mapping[Axes, int]:
        return self._shape

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        return np.full(
            shape=(self.shape[Axes.Y], self.shape[Axes.X]),
            fill_value=1.0,
            dtype=np.float32)


def tile_fetcher_factory(
        fetched_tile_cls: Type[FetchedTile],
        pass_tile_indices: bool=False,
        *fetched_tile_constructor_args,
        **fetched_tile_constructor_kwargs,
) -> TileFetcher:
    """
    Given a class of that implements :class:`.FetchedTile`, return a TileFetcher that returns an
    instance of that class.  If `pass_tile_indices` is True, then the TileFetcher is constructed
    with fov/round/ch/z.  The constructor is always invoked with variable-length arguments from
    `fetched_tile_constructor_args` and keyword arguments from `fetched_tile_constructor_kwargs`.
    """
    class ResultingClass(TileFetcher):
        def get_tile(
                self, fov_id: int, round_label: int, ch_label: int, zplane_label: int,
        ) -> FetchedTile:
            args = list()
            if pass_tile_indices:
                args.extend([fov_id, round_label, ch_label, zplane_label])
            args.extend(fetched_tile_constructor_args)

            return fetched_tile_cls(*args, **fetched_tile_constructor_kwargs)

    return ResultingClass()
