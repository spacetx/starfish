class DeprecatedAPIError(Exception):
    """
    Raised when using an API that has been deprecated.
    """
    pass


class DataFormatWarning(Warning):
    """
    Warnings given by starfish when the data is not formatted as expected, though not fatally.
    """
    pass
