from enum import Enum


class Coordinates(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, Coordinates) or isinstance(other, str):
            return self.value == other
        return False


class Indices(Enum):
    HYB = 'h'
    CH = 'c'
    Z = 'z'

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, Indices) or isinstance(other, str):
            return self.value == other
        return False
