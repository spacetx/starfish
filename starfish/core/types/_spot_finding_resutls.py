import random
from typing import Dict, List, Tuple

from starfish.core.types import SpotAttributes


class SpotFindingResults:

    def __init__(self, spot_attributes_list: List[Tuple] = None):
        if spot_attributes_list:
            for indices, spots in spot_attributes_list:
                self._results[indices] = spots
        else:
            self._results: Dict[Tuple[int, int], SpotAttributes] = dict()

    def __setitem__(self, axes, spots):
        self._results[axes] = spots

    def __getitem__(self, key):
        return self._results[key]

    def __len__(self):
        return len(self._results)

    def indices(self):
        return self._results.keys()

    def all_spots(self):
        return self._results.values()

    def round_labels(self):
        return list(set(sorted(r for (r, ch) in self.indices())))

    def ch_labels(self):
        return list(set(sorted(ch for (r, ch) in self.indices())))
