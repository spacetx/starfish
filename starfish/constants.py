from enum import Enum


class AugmentedEnum(Enum):
    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, type(self)) or isinstance(other, str):
            return self.value == other
        return False

    def __str__(self) -> str:
        return self.value


class Coordinates(AugmentedEnum):
    Z = 'z'
    Y = 'y'
    X = 'x'


class Indices(AugmentedEnum):
    ROUND = 'r'
    CH = 'c'
    Z = 'z'


class Features:
    """
    contains constants relating to the codebook and feature (spot/pixel) representations of the
    image data
    """

    AXIS = 'features'
    TARGET = 'target'
    CODEWORD = 'codeword'
    CODE_VALUE = 'v'
    SPOT_RADIUS = 'radius'
    DISTANCE = 'distance'
    Z = 'z'
    Y = 'y'
    X = 'x'
