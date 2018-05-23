import os

import collections
from typing import Tuple

from slicedimage import ImageFormat
from slicedimage.io import resolve_url

from ._base import ImageBase


class SingleImage(ImageBase):
    def __init__(self, data):
        self._data = data

    @classmethod
    def from_url(cls, relativeurl: str, baseurl: str):
        backend, name, baseurl = resolve_url(relativeurl, baseurl)
        extension = os.path.splitext(relativeurl)[1].lstrip(".")
        format = ImageFormat.find_by_extension(extension)
        with backend.read_file_handle(name) as fh:
            return cls(format.reader_func(fh))

    @property
    def numpy_array(self):
        data = self._data.view()
        data.setflags(write=False)
        return data

    @numpy_array.setter
    def numpy_array(self, data):
        self._data = data.view()
        data.setflags(write=False)

    @property
    def raw_shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def shape(self) -> collections.OrderedDict:
        """
        Returns the shape of the space that this image inhabits.  It does not include the dimensions of the image
        itself.  For instance, if this is an X-Y image in a C-H-Y-X space, then the shape would include the dimensions C
        and H.

        Note that for SingleImage, this is going to be an empty dict, as there are no extrinsic dimensions.
        """
        return collections.OrderedDict()

    def write(self, filepath):
        import numpy
        numpy.save(filepath, self._data)

    def max_proj(self, *dims):
        """
        Returns max projection of this image across one of the dimensions extrinsic to the image.  For instance, if this
        is an X-Y image in a C-H-Y-X space, then one can take the max projection across the combinations of C and H.

        Note that for SingleImage, it should always return the stored image exactly, as there should be no extrinsic
        dimensions.
        """
        return self.numpy_array
