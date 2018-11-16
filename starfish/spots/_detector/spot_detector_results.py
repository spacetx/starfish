from typing import List, NamedTuple, Optional, Union

import numpy as np


class LocalMaxFinderResults(NamedTuple):
    thresholds: Optional[np.ndarray]
    spot_count: Optional[List[int]]
    grad: int
    spop_props: list
    labels: Union[np.ndarray, int]