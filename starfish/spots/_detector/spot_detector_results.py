from typing import List, NamedTuple, Optional, Union

import numpy as np


class LocalMaxFinderResults(NamedTuple):
    thresholds: Optional[np.ndarray]
    spot_count: Optional[List[int]]
    grad: Optional[int]
    spot_props: Optional[list]
    labels: Union[np.ndarray, int]
