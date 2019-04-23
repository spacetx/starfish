import abc
from typing import Generic, Iterable, TypeVar

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.util.indirectfile import (
    ConversionFormula,
    convert,
    GetCodebook,
    GetCodebookFromExperiment,
    GetImageStack,
    GetImageStackFromExperiment,
    NoApplicableConversionFormulaError,
    NoSuccessfulConversionFormulaError,
)
from . import ParamType


IndirectResultType = TypeVar("IndirectResultType")


class IndirectFile(ParamType, Generic[IndirectResultType]):
    def convert(self, value: str, param, ctx):
        conversion_formulas = self.get_conversion_formulas()
        try:
            return convert(value, conversion_formulas)
        except (NoApplicableConversionFormulaError, NoSuccessfulConversionFormulaError) as ex:
            self.fail(ex.args[0])

    @abc.abstractmethod
    def get_conversion_formulas(self) -> Iterable[ConversionFormula[IndirectResultType]]:
        """Return one or more conversion Formulas to get from an input string to the type of object
        we want.
        """
        raise NotImplementedError()


class CodebookParam(IndirectFile[Codebook]):
    def __init__(self):
        self.name = "codebook"

    def get_conversion_formulas(self) -> Iterable[ConversionFormula[Codebook]]:
        return [
            GetCodebookFromExperiment(),
            GetCodebook(),
        ]


CodebookParamType = CodebookParam()


class ImageStackParam(IndirectFile[ImageStack]):
    def __init__(self):
        self.name = "imagestack"

    def get_conversion_formulas(self) -> Iterable[ConversionFormula[ImageStack]]:
        return [
            GetImageStackFromExperiment(),
            GetImageStack(),
        ]


ImageStackParamType = ImageStackParam()
