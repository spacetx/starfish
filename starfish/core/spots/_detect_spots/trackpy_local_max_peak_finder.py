import warnings
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from trackpy import locate

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, SpotAttributes
from starfish.core.util import click
from ._base import DetectSpotsAlgorithmBase
from .detect import detect_spots


class TrackpyLocalMaxPeakFinder(DetectSpotsAlgorithmBase):
    """
    Find spots using a local max peak finding algorithm

    This is a wrapper for :code:`trackpy.locate`, which implements a version of the
    `Crocker-Grier <crocker_grier>`_ algorithm.

    .. _crocker_grier: https://physics.nyu.edu/grierlab/methods3c/

    Parameters
    ----------

    spot_diameter : odd integer or tuple of odd integers.
        This may be a single number or a tuple giving the feature’s extent in each dimension,
        useful when the dimensions do not have equal resolution (e.g. confocal microscopy).
        The tuple order is the same as the image shape, conventionally (z, y, x) or (y, x).
        The number(s) must be odd integers. When in doubt, round up.
    min_mass : float, optional
        The minimum integrated brightness. This is a crucial parameter for eliminating spurious
        features. Recommended minimum values are 100 for integer images and 1 for float images.
        Defaults to 0 (no filtering).
    max_size : float
        maximum radius-of-gyration of brightness, default None
    separation : float or tuple
        Minimum separtion between features. Default is diameter + 1. May be a tuple, see
        diameter for details.
    percentile : float
        Features must have a peak brighter than pixels in this percentile. This helps eliminate
        spurious peaks.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels Default is 1. May be a tuple, see diameter
        for details.
    smoothing_size : float or tuple
        The size of the sides of the square kernel used in boxcar (rolling average) smoothing,
        in pixels Default is diameter. May be a tuple, making the kernel rectangular.
    threshold : float
        Clip bandpass result below this value. Thresholding is done on the already
        background-subtracted image. By default, 1 for integer images and 1/255 for float
        images.
    measurement_type : str ['max', 'mean']
        name of the function used to calculate the intensity for each identified spot area
    preprocess : boolean
        Set to False to turn off bandpass preprocessing.
    max_iterations : integer
        Max number of loops to refine the center of mass, default 10
    is_volume : bool
        if True, run the algorithm on 3d volumes of the provided stack
    verbose : bool
        If True, report the percentage completed (default = False) during processing

    Notes
    -----
    See also trackpy locate: http://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html

    """

    def __init__(
            self, spot_diameter, min_mass, max_size, separation, percentile=0,
            noise_size: Tuple[int, int, int] = (1, 1, 1), smoothing_size=None, threshold=None,
            preprocess: bool = False, max_iterations: int = 10, measurement_type: str = 'max',
            is_volume: bool = False, verbose=False) -> None:

        self.diameter = spot_diameter
        self.minmass = min_mass
        self.maxsize = max_size
        self.separation = separation
        self.noise_size = noise_size
        self.smoothing_size = smoothing_size
        self.percentile = percentile
        self.threshold = threshold
        self.measurement_function = self._get_measurement_function(measurement_type)
        self.preprocess = preprocess
        self.max_iterations = max_iterations
        self.is_volume = is_volume
        self.verbose = verbose

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """

        Parameters
        ----------
        data_image : np.ndarray
            three-dimensional image containing spots to be detected

        Returns
        -------
        SpotAttributes :
            spot attributes table for all detected spots

        """
        data_image = np.asarray(data_image)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)  # trackpy numpy indexing warning
            warnings.simplefilter('ignore', UserWarning)  # yielded if black images
            attributes = locate(
                data_image,
                diameter=self.diameter,
                minmass=self.minmass,
                maxsize=self.maxsize,
                separation=self.separation,
                noise_size=self.noise_size,
                smoothing_size=self.smoothing_size,
                threshold=self.threshold,
                percentile=self.percentile,
                preprocess=self.preprocess,
                max_iterations=self.max_iterations,
            )

        # when zero spots are detected, 'ep' is missing from the trackpy locate results.
        if attributes.shape[0] == 0:
            attributes['ep'] = []

        # TODO ambrosejcarr: data should always be at least pseudo-3d, this may not be necessary
        # TODO ambrosejcarr: this is where max vs. sum vs. mean would be parametrized.
        # here, total_intensity = sum, intensity = max
        new_colnames = [
            'y', 'x', 'total_intensity', 'radius', 'eccentricity', 'intensity', 'raw_mass', 'ep'
        ]
        if len(data_image.shape) == 3:
            attributes.columns = ['z'] + new_colnames
        else:
            attributes.columns = new_colnames

        attributes['spot_id'] = np.arange(attributes.shape[0])
        return SpotAttributes(attributes)

    def run(
            self,
            primary_image: ImageStack,
            blobs_image: Optional[ImageStack] = None,
            blobs_axes: Optional[Tuple[Axes, ...]] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> IntensityTable:
        """
        Find spots.

        Parameters
        ----------
        primary_image : ImageStack
            ImageStack where we find the spots in.
        blobs_image : Optional[ImageStack]
            If provided, spots will be found in the blobs image, and intensities will be measured
            across rounds and channels. Otherwise, spots are measured independently for each channel
            and round.
        blobs_axes : Optional[Tuple[Axes, ...]]
            If blobs_image is provided, blobs_axes must be provided as well.  blobs_axes represents
            the axes across which the blobs image is max projected before spot detection is done.
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """
        intensity_table = detect_spots(
            data_stack=primary_image,
            spot_finding_method=self.image_to_spots,
            reference_image=blobs_image,
            reference_image_max_projection_axes=blobs_axes,
            measurement_function=self.measurement_function,
            n_processes=n_processes,
            radius_is_gyration=True)

        return intensity_table

    @staticmethod
    @click.command("TrackpyLocalMaxPeakFinder")
    @click.option("--spot-diameter", type=str, help='expected spot size')
    @click.option(
        "--min-mass", default=4, type=int, help="minimum integrated spot intensity")
    @click.option(
        "--max-size", default=6, type=int, help="maximum radius of gyration of brightness")
    @click.option(
        "--separation", default=5, type=float, help="minimum distance between spots")
    @click.option(
        "--noise-size", default=None, type=int,
        help="width of gaussian blurring kernel, in pixels")
    @click.option(
        "--smoothing-size", default=None, type=int,
        help="odd integer. Size of boxcar (moving average) filter in pixels. Default is the "
             "Diameter")
    @click.option(
        "--preprocess", is_flag=True,
        help="if passed, gaussian and boxcar filtering are applied")
    @click.option(
        "--max-iterations", default=10, type=int,
        help="Max number of loops to refine the center of mass. Default is 10")
    @click.option(
        "--show", default=False, is_flag=True, help="display results visually")
    @click.option(
        "--percentile", default=None, type=float,
        help="clip bandpass below this value. Thresholding is done on already background-"
             "subtracted images. Default 1 for integer images and 1/255 for float")
    @click.option(
        "--is-volume", is_flag=True,
        help="indicates that the image stack should be filtered in 3d")
    @click.pass_context
    def _cli(ctx, spot_diameter, min_max, max_size, separation, noise_size, smoothing_size,
             preprocess, max_iterations, show, percentile, is_volume):

        instance = TrackpyLocalMaxPeakFinder(spot_diameter, min_max, max_size,
                                             separation, noise_size, smoothing_size,
                                             preprocess, max_iterations, show,
                                             percentile, is_volume)
        ctx.obj["component"]._cli_run(ctx, instance)
