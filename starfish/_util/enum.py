from typing import Iterable, List, Union

from starfish.types import Axes

def harmonize(iterable: Iterable[Union[Axes, str]]) -> List[str]:
    return list(str(v) for v in iterable)
