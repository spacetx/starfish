import argparse
import multiprocessing
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Indices
from ._base import FilterAlgorithmBase


class ZeroByChannelMagnitude(FilterAlgorithmBase):
    def __init__(self, thresh: int, normalize: bool, is_volume: bool=False, **kwargs) -> None:
        """For assays in which we expect codewords to have explicit zero values,
        e.g., DARTFISH, SEQFISH, etc., this filter allows for the explicit zeroing
        out of pixels, for each round, where there is insufficient signal magnitude across channels.


        Parameters
        ----------
        thresh : int
            pixels in each round that have a L2 norm across channels below this threshold
            are set to 0
        is_volume : bool = False
            Currently, only 2d-data is supported  # TODO dganguli: please implement 3D

        normalize : bool
            if True, this scales all rounds to have unit L2 norm across channels
        """
        self.thresh = thresh
        self.normalize = normalize
        self.is_volume = is_volume

    @classmethod
    def _add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            '--thresh', type=float,
            help='minimum magnitude threshold for pixels across channels')
        group_parser.add_argument(
            '--normalize', action="store_true",
            help='Scales all rounds to have unit L2 norm across channels')

    @staticmethod
    def zero_by_channel_magnitude(r_dat: Tuple[int, xr.DataArray], thresh, stack, normalize):
        # nervous about how xarray orders dimensions so i put this here explicitly ....
        r, dat = r_dat
        dat = dat.transpose(Indices.CH.value,
                            Indices.Z.value,
                            Indices.Y.value,
                            Indices.X.value
                            )
        # ... to account for this line taking the norm across axis 0, or the channel axis
        ch_magnitude = np.linalg.norm(dat, ord=2, axis=0)
        magnitude_mask = ch_magnitude >= thresh

        # apply mask and optionally, normalize by channel magnitude
        for c in range(stack.num_chs):
            ind = {Indices.ROUND.value: r, Indices.CH.value: c}
            stack._data[ind] = stack._data[ind] * magnitude_mask

            if normalize:
                stack._data[ind] = np.divide(stack._data[ind],
                                             ch_magnitude,
                                             where=magnitude_mask
                                             )

    def run(
            self, stack: ImageStack, in_place: bool=True, verbose=False,
            n_processes: Optional[int]=None
    ) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on the percentage completed during processing (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack
            Contains filtered data. If in_place is True, returns a reference to the input stack

        """

        if not in_place:
            new_stack = deepcopy(stack)
            return self.run(new_stack, in_place=True)

        channels_per_round = stack._data.groupby(Indices.ROUND.value)
        channels_per_round = tqdm(channels_per_round) if verbose else channels_per_round
        mapfunc = partial(
            self.zero_by_channel_magnitude,
            thresh=self.thresh, stack=stack, normalize=self.normalize)

        p = multiprocessing.Pool(n_processes)
        # compute channel magnitude mask
        p.imap_unordered(mapfunc, channels_per_round)

        p.close()
        p.join()

        return stack
