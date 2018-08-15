import argparse
from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from starfish.constants import Indices, Coordinates
from starfish.stack import ImageStack
from ._base import FilterAlgorithmBase


class ZeroByChannelMagnitude(FilterAlgorithmBase):
    def __init__(self, thresh: int, normalize: bool, **kwargs) -> None:
        """For assays in which we expect codewords to have explicit zero values,
        e.g., DARTFISH, SEQFISH, etc., this filter allows for the explicit zeroing
        out of pixels, for each round, where there is insufficient signal magnitude across channels.


        Parameters
        ----------
        thresh : int
            pixels in each round that have a L2 norm across channels below this threshold
            are set to 0

        normalize : bool
            if True, this scales all rounds to have unit L2 norm across channels
        """
        self.thresh = thresh
        self.normalize = normalize

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            '--thresh', type=int, help='minimum magnitude threshold for pixels across channels')

    def run(self, stack: ImageStack, in_place: bool = True, verbose=False) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on the percentage completed during processing (default = False)

        Returns
        -------
        Optional[ImageStack] :
            if in-place is False, return the results of filter as a new stack

        """

        channels_per_round = stack._data.groupby(Indices.ROUND.value)
        channels_per_round = tqdm(channels_per_round) if verbose else channels_per_round

        if not in_place:
            new_stack = deepcopy(stack)
            return self.run(new_stack, in_place=True)

        # compute channel magnitude mask
        for r, dat in channels_per_round:
            # nervous about how xarray orders dimensions so i put this here explicitly ....
            dat = dat.transpose(Indices.CH.value,
                                Indices.Z.value,
                                Coordinates.Y.value,
                                Coordinates.X.value
                                )
            # ... to account for this line taking the norm across axis 0, or the channel axis
            ch_magnitude = np.linalg.norm(dat, ord=2, axis=0)
            magnitude_mask = ch_magnitude >= self.thresh

            # apply mask and optionally, normalize by channel magnitude
            for c in range(stack.num_chs):
                ind = {Indices.ROUND: r, Indices.CH: c}
                stack._data[ind] = stack._data[ind] * magnitude_mask

                if self.normalize:
                    stack._data[ind] = np.divide(stack._data[ind],
                                                 ch_magnitude,
                                                 where=magnitude_mask
                                                 )

        return stack

# TODO @ajc the code below fulfills the same task without xarray.
# instead it uses vectorized numpy array. what do we prefer?

# def zero_channels_old(stack, magnitude_thresh, normalize_by_magnitude=False):
#     """
#     Sets all values in a round to zero if the magnitude across color channel is below a
#     specified threshold
#     """
#     # r, z, y, x
#     ch_magnitudes = np.linalg.norm(stack.numpy_array, ord=2, axis=1)
#     magnitude_mask = ch_magnitudes >= magnitude_thresh
#
#     magnitude_mask_repeat = np.expand_dims(magnitude_mask, axis=1)
#     magnitude_mask_repeat = magnitude_mask_repeat.repeat(stack.num_chs, axis=1)
#
#     ch_magnitudes_repeat = np.expand_dims(ch_magnitudes, axis=1)
#     ch_magnitudes_repeat = ch_magnitudes_repeat.repeat(stack.num_chs, axis=1)
#
#     res = stack.numpy_array * magnitude_mask_repeat
#
#     if normalize_by_magnitude:
#         res = np.divide(stack.numpy_array, ch_magnitudes_repeat, where=magnitude_mask_repeat)
#
#     return ImageStack.from_numpy_array(res), ch_magnitudes
