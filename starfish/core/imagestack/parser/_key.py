import typing

from starfish.core.types import Axes


class TileKey:
    """
    This class is used to index into the TileData class.
    """
    def __init__(self, *, round: int, ch: int, zplane: int) -> None:
        self._round = round
        self._ch = ch
        self._zplane = zplane

    @property
    def round(self) -> int:
        return self._round

    @property
    def ch(self) -> int:
        return self._ch

    @property
    def z(self) -> int:
        return self._zplane

    AXES_TO_PROPERTY_MAP = {
        Axes.ROUND: round,
        Axes.CH: ch,
        Axes.ZPLANE: z,
    }

    def __getitem__(self, item) -> int:
        """Given a index, return the corresponding value for that index for this tilekey.  For
        instance, tilekey[Axes.ROUND] returns the round for this tilekey."""
        index = typing.cast(Axes, item)
        prop: property = TileKey.AXES_TO_PROPERTY_MAP[index]  # type: ignore
        return prop.__get__(self, TileKey)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TileKey):
            return False

        return self._round == other.round and self._ch == other.ch and self._zplane == other.z

    def __hash__(self) -> int:
        return int(self._round ^ self._ch ^ self._zplane)

    def __repr__(self) -> str:
        return f"(round: {self._round} ch: {self._ch} z: {self._zplane})"
