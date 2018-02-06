import enum

import numpy
import skimage.io


class ImageBase(object):
    @property
    def numpy_array(self):
        """Retrieves the image data as a numpy array."""
        raise NotImplementedError()

    @property
    def shape(self):
        """Retrieves the shape of the image data."""
        raise NotImplementedError()

    def write(self, filepath):
        """
        Writes the image out to disk.

        Args:
            filepath: The path of the file to write the image to.
        """
        raise NotImplementedError()


class ImageFormat(enum.Enum):
    TIFF = (skimage.io.imread, "tiff")
    NUMPY = (numpy.load, "npy")

    def __init__(self, reader_func, file_ext):
        self._reader_func = reader_func
        self._file_ext = file_ext

    @property
    def reader_func(self):
        return self._reader_func

    @property
    def file_ext(self):
        return self._file_ext
