import collections
from typing import Optional, Sequence, Tuple

from numpy import ndarray

from starfish.constants import Indices


class ImageBase(object):
    @classmethod
    def from_url(cls, url: str, baseurl: Optional[str]) -> "ImageBase":
        """
        Constructs an ImageBase object from a URL and a base URL.

        Parameters:
        -----------
        url : str
            Either an absolute URL or a relative URL referring to the image to be read.
        baseurl : Optional[str]
            If url is a relative URL, then this must be provided.  If url is an absolute URL, then this parameter is
            ignored.
        """
        raise NotImplementedError()

    @property
    def numpy_array(self) -> ndarray:
        """Retrieves a view of the image data as a numpy array."""
        raise NotImplementedError()

    @numpy_array.setter
    def numpy_array(self, data: ndarray):
        """Sets the image's data from a numpy array.  The numpy array is advised to be immutable afterwards."""
        raise NotImplementedError()

    @property
    def raw_shape(self) -> Tuple[int]:
        """
        Returns the shape of the space that this image inhabits.  It does not include the dimensions of the image
        itself.  For instance, if this is an X-Y image in a C-H-Y-X space, then the shape would include the dimensions C
        and H.

        Returns
        -------
        Tuple[int] :
            The sizes of the indices.
        """
        raise NotImplementedError()

    @property
    def shape(self) -> collections.OrderedDict:
        """
        Returns the shape of the space that this image inhabits.  It does not include the dimensions of the image
        itself.  For instance, if this is an X-Y image in a C-H-Y-X space, then the shape would include the dimensions C
        and H.

        Returns
        -------
        An ordered mapping between index names to the size of the index.
        """
        raise NotImplementedError()

    def write(self, filepath: str) -> None:
        """
        Writes the image out to disk.

        Parameters:
        -----------
        filepath : str
            The path of the file to write the image to.
        """
        raise NotImplementedError()

    def max_proj(self, *dims: Sequence[Indices]) -> ndarray:
        """
        Returns max projection of this image across one of the dimensions extrinsic to the image.  For instance, if this
        is an X-Y image in a C-H-Y-X space, then one can take the max projection across the combinations of C and H.
        """
        raise NotImplementedError()
