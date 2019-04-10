import enum
from typing import Any, Callable, Type

from starfish.codebook.codebook import Codebook
from starfish.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.util.indirectfile import (
    convert,
    GetCodebook,
    GetCodebookFromExperiment,
    GetImageStack,
    GetImageStackFromExperiment,
)


def imagestack_convert(indirect_path_or_url: str) -> ImageStack:
    return convert(
        indirect_path_or_url,
        [
            GetImageStack(),
            GetImageStackFromExperiment(),
        ],
    )


def codebook_convert(indirect_path_or_url: str) -> Codebook:
    return convert(
        indirect_path_or_url,
        [
            GetCodebook(),
            GetCodebookFromExperiment(),
        ],
    )


class FileTypes(enum.Enum):
    """These are the filetypes supported as inputs and outputs for recipes.  Each filetype is
    associated with the implementing class, the method to invoke to load such a filetype, and the
    method to invoke to save back to the filetype.

    The load method is expected to be called with a string, which is the file or url to load from,
    and is expected to return an instantiated object.

    The save method is expected to be called with the object and a string, which is the path to
    write the object to.
    """
    IMAGESTACK = (ImageStack, imagestack_convert, ImageStack.export)
    INTENSITYTABLE = (IntensityTable, IntensityTable.load, IntensityTable.save)
    EXPRESSIONMATRIX = (ExpressionMatrix, ExpressionMatrix.load, ExpressionMatrix.save)
    CODEBOOK = (Codebook, codebook_convert, Codebook.to_json)

    def __init__(self, cls: Type, loader: Callable[[str], Any], saver: Callable[[Any, str], None]):
        self._cls = cls
        self._load = loader
        self._save = saver

    @property
    def load(self) -> Callable[[str], Any]:
        return self._load

    @property
    def save(self) -> Callable[[Any, str], None]:
        return self._save

    @staticmethod
    def resolve_by_class(cls: Type) -> "FileTypes":
        for member in FileTypes.__members__.values():
            if cls == member.value[0]:
                return member
        raise TypeError(f"filetype {cls} not supported.")

    @staticmethod
    def resolve_by_instance(instance) -> "FileTypes":
        for member in FileTypes.__members__.values():
            if isinstance(instance, member.value[0]):
                return member
        raise TypeError(f"filetype of {instance.__class__} not supported.")


class FileProvider:
    """This is used to wrap paths or URLs that are passed into Runnables via the `file_inputs` magic
    variable.  This is so we can differentiate between strings and `file_inputs` values, which must
    be first constructed into a starfish object via its loader."""
    def __init__(self, path_or_url: str) -> None:
        self.path_or_uri = path_or_url

    def __str__(self):
        return f"FileProvider(\"{self.path_or_uri}\")"


class TypedFileProvider:
    """Like :py:class:`FileProvider`, this is used to wrap paths or URLs that are passed into
    Runnables via the `file_inputs` magic variable.  In this case, the object type has been
    resolved by examining the type annotation."""
    def __init__(self, backing_file_provider: FileProvider, object_class: Type) -> None:
        self.backing_file_provider = backing_file_provider
        self.type = FileTypes.resolve_by_class(object_class)

    def load(self) -> Any:
        return self.type.load(self.backing_file_provider.path_or_uri)
