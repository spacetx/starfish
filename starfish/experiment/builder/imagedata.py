from io import BytesIO
from typing import IO, Tuple

import numpy as np
from skimage.io import imsave
from slicedimage import (
    ImageFormat,
)


class FetchedImage:
    """
    This is the contract for providing the data for constructing a :class:`slicedimage.Tile`.
    """
    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def format(self) -> ImageFormat:
        raise NotImplementedError()

    @property
    def image_data_handle(self) -> IO:
        raise NotImplementedError()


class ImageFetcher:
    """
    This is the contract for providing the image data for constructing a
    :class:`slicedimage.Collection`.
    """
    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        """
        Given fov, hyb, ch, and z, return an instance of a :class:`.FetchedImage` that can be
        queried for the image data.
        """
        raise NotImplementedError()


class RandomNoiseImage(FetchedImage):
    """
    This is a simple implementation of :class:`.FetchedImage` that simply regenerates random data
    for the image.
    """
    @property
    def shape(self) -> Tuple[int, ...]:
        return 1536, 1024

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    def image_data_handle(self) -> IO:
        arr = np.random.randint(0, 256, size=self.shape, dtype=np.uint8)
        output = BytesIO()
        imsave(output, arr, plugin="tifffile")
        output.seek(0)
        return output


class RandomNoiseImageFetcher(ImageFetcher):
    """
    This is a simple implementation of :class:`.ImageFetcher` that simply returns a
    :class:`.RandomNoiseImage` for every fov, hyb, ch, z combination.
    """
    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        return RandomNoiseImage()
