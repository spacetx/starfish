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
