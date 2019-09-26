from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from starfish.core.config import StarfishConfig
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from ._base import FilterAlgorithm


class ZeroByChannelMagnitude(FilterAlgorithm):
    """
    For assays in which we expect codewords to have explicit zero values,
    e.g., DARTFISH, this filter allows for the explicit zeroing
    out of pixels, for each round, where there is insufficient signal magnitude across channels.

    Parameters
    ----------
    thresh : int
        pixels in each round that have a L2 norm across channels below this threshold
        are set to 0
    normalize : bool
        if True, this scales all rounds to have unit L2 norm across channels
    """

    def __init__(self, thresh: int, normalize: bool) -> None:

        self.thresh = thresh
        self.normalize = normalize

    _DEFAULT_TESTING_PARAMETERS = {"thresh": 0, "normalize": True}

    def run(
            self, stack: ImageStack,
            in_place: bool = False,
            verbose=False,
            n_processes: Optional[int] = None,
            *args,
    ) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on the percentage completed during processing (default = False)
        n_processes : Optional[int]: None
            Not implemented. Number of processes to use when applying filter.

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        # The default is False, so even if code requests True require config to be True as well
        verbose = verbose and StarfishConfig().verbose
        channels_per_round = stack.xarray.groupby(Axes.ROUND.value)
        channels_per_round = tqdm(channels_per_round) if verbose else channels_per_round

        if not in_place:
            new_stack = deepcopy(stack)
            self.run(new_stack, in_place=True)
            return new_stack

        # compute channel magnitude mask
        for r, dat in channels_per_round:
            # nervous about how xarray orders dimensions so i put this here explicitly ....
            dat = dat.transpose(Axes.CH.value,
                                Axes.ZPLANE.value,
                                Axes.Y.value,
                                Axes.X.value
                                )
            # ... to account for this line taking the norm across axis 0, or the channel axis
            ch_magnitude = np.linalg.norm(dat, ord=2, axis=0)
            magnitude_mask = ch_magnitude >= self.thresh

            # apply mask and optionally, normalize by channel magnitude
            for c in stack.axis_labels(Axes.CH):
                ind = {Axes.ROUND.value: r, Axes.CH.value: c}
                stack._data[ind] = stack._data[ind] * magnitude_mask

                if self.normalize:
                    stack._data[ind] = np.divide(stack._data[ind],
                                                 ch_magnitude,
                                                 where=magnitude_mask
                                                 )
        return None
