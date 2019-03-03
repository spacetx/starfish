import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes, Features
from starfish.util import click
from ._base import FilterAlgorithmBase

# TODO ambrosejcarr replace this with constitutive channel fluorescence mask when that PR is merged
def _generate_test_codebook():
    n_channel = 2
    n_round = 2
    data = np.zeros((n_channel, n_channel, n_round))
    for c in range(n_channel):
        data[c, c] = 1
        code_names = [str(c) for c in range(n_channel)]
    return Codebook._create_codebook(code_names, n_channel, n_round, data)


class Decode(FilterAlgorithmBase):

    def __init__(self, codebook: Codebook, metric: str="euclidean", norm_order: int=2) -> None:
        """Decode an image in round/channel space to target space

        The resulting ImageStack will be of shape (1, n_targets, z, y, x) where n_targets are the
        number of targets in the codebook. The values of the decoded images are the distance of
        each pixel from the given target, calculated using the provided distance.

        Parameters
        ----------
        codebook : Codebook
            codebook that will be used to decode image
        metric : str
            the metric used to calculate pixel intensity distance from codes in codebook

        """
        self.codebook = codebook
        self.metric = metric
        self.norm_order = norm_order

    _DEFAULT_TESTING_PARAMETERS: dict = {
        "codebook": _generate_test_codebook()
    }

    @staticmethod
    def _decode(
        image: xr.DataArray,
        codebook: Codebook,
        metric: str="euclidean",
        norm_order: int=2
    ) -> np.ndarray:
        """
        """
        traces = image.xarray.stack(
            features=(Axes.ZPLANE.value, Axes.Y.value, Axes.X.value)
        )
        traces = traces.transpose(Features.AXIS, Axes.CH.value, Axes.ROUND.value)

        # normalize codes and traces
        norm_intensities, norms = codebook._normalize_features(traces, norm_order=norm_order)
        norm_codes, _ = codebook._normalize_features(codebook, norm_order=norm_order)

        intensity_traces = norm_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
        intensity_codes = norm_codes.stack(traces=(Axes.CH.value, Axes.ROUND.value))
        distances = cdist(intensity_traces, intensity_codes)

        # reshape into an ImageStack
        distances = xr.DataArray(distances, coords=(norm_intensities.features, norm_codes.target))
        distances = distances.unstack(Features.AXIS)
        distances = distances.rename(target=Axes.CH.value)
        distance_image = distances.expand_dims(Axes.ROUND.value)

        # normalize distances into (0, 1) to keep ImageStack requirements
        normalized_image = distance_image / distance_image.max()

        # TODO return a label image
        return ImageStack.from_numpy_array(normalized_image.values)

    def run(self, stack, in_place=False, verbose=False, n_processes=None) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            Not used. Decode changes the shape of the input image.
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        # TODO support multiprocessing
        return self._decode(stack, self.codebook, self.metric)

    @staticmethod
    @click.command("Decode")
    @click.option(
        "--codebook", type=str, required=True, help="Codebook used to decode images")
    @click.option(
        "--metric", default="euclidean", type=str, help="distance metric")
    @click.option(
        "--n_codes", type=int, default=1, help="if True, return only closest target for each pixel")
    @click.pass_context
    def _cli(ctx, codebook, metric, n_codes):
        codebook_obj = Codebook.from_json(codebook)
        ctx.obj["component"]._cli_run(ctx, Decode(codebook_obj, metric, n_codes))
