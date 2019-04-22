import abc
from typing import Generic, Iterable, TypeVar


FormulaResultType = TypeVar("FormulaResultType")


class ConversionFormula(Generic[FormulaResultType]):
    """A conversion formula represents a plausible contract to convert a string value to another
    data type.  Each conversion formula implements two methods -- a lightweight method to determine
    if this formula can be applied given a string value, and a load method that actually does the
    conversion.
    """
    @abc.abstractmethod
    def applicable(self, input_parameter: str) -> bool:
        """Returns true iff this formula might work for the given input."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, input_parameter: str) -> FormulaResultType:
        """Attempt to run this conversion formula against this input."""
        raise NotImplementedError()


class NoApplicableConversionFormulaError(Exception):
    """Raised when no conversion formula declared itself applicable to this input string."""
    pass


class NoSuccessfulConversionFormulaError(Exception):
    """Raised when all the conversion formulas that declared itself applicable failed to execute
    successfully."""
    pass


ConvertResultType = TypeVar("ConvertResultType")


def convert(
        value: str,
        conversion_formulas: Iterable[ConversionFormula[ConvertResultType]]) -> ConvertResultType:
    """
    Given a string value and a series of conversion formulas, attempt to convert the value using the
    formulas.

    If none of the formulas declare themselves as applicable, then raise
    :py:class:`NoApplicableConversionformulaError`.  If none of the formulas that declare themselves
    as eligible run successfully, then raise :py:class:`NoSuccessfulConversionformulaError`.

    Parameters
    ----------
    value : str
        The string value we are attempting to convert.
    conversion_formulas : Iterable[ConversionFormula[ConvertResultType]]
        A series of conversion formulas.

    Returns
    -------
    The converted value.
    """
    none_applied = True

    for conversion_formula in conversion_formulas:
        if conversion_formula.applicable(value):
            none_applied = False
            try:
                return conversion_formula.load(value)
            except Exception:
                pass

    if none_applied:
        raise NoApplicableConversionFormulaError(
            f"Could not find applicable gonversion formula for {value}")
    raise NoSuccessfulConversionFormulaError(
        f"All applicable conversion formulas failed to run successfully for {value}.")
