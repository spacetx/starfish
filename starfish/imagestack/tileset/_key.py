class TileKey:
    """
    This class is used to index into the TileSetMetadata class.
    """
    def __init__(self, *, round: int, ch: int, z: int) -> None:
        self._round = round
        self._ch = ch
        self._z = z

    @property
    def round(self) -> int:
        return self._round

    @property
    def ch(self) -> int:
        return self._ch

    @property
    def z(self) -> int:
        return self._z

    def __eq__(self, other) -> bool:
        if not isinstance(other, TileKey):
            return False

        return self._round == other.round and self._ch == other.ch and self._z == other.z

    def __hash__(self) -> int:
        return int(self._round ^ self._ch ^ self._z)

    def __repr__(self) -> str:
        return f"(round: {self._round} ch: {self._ch} z: {self._z})"
