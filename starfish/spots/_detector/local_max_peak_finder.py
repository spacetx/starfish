from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from trackpy import locate

from starfish.intensity_table import IntensityTable
from starfish.stack import ImageStack
from starfish.types import SpotAttributes
from ._base import SpotFinderAlgorithmBase
from .detect import detect_spots


class LocalMaxPeakFinder(SpotFinderAlgorithmBase):

    def __init__(
            self, spot_diameter, min_mass, max_size, separation, percentile=0,
            noise_size: Tuple[int, int, int]=(1, 1, 1), smoothing_size=None, threshold=None,
            preprocess: bool=False, measurement_type: str='max', is_volume: bool=False,
            verbose=False, **kwargs) -> None:
        """Find spots using a local max peak finding algorithm

        This is a wrapper for `trackpy.locate`

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
            max number of loops to refine the center of mass, default 10
        is_volume : bool
            if True, run the algorithm on 3d volumes of the provided stack
        verbose : bool
            If True, report the percentage completed (default = False) during processing


        See Also
        --------
        trackpy locate: http://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html

        """

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
        self.is_volume = is_volume
        self.verbose = verbose

    def image_to_spots(self, image: np.ndarray) -> SpotAttributes:
        """

        Parameters
        ----------
        image : np.ndarray
            three-dimensional numpy array containing spots to detect

        Returns
        -------
        SpotAttributes :
            spot attributes table for all detected spots

        """
        attributes = locate(
            image,
            diameter=self.diameter,
            minmass=self.minmass,
            maxsize=self.maxsize,
            separation=self.separation,
            noise_size=self.noise_size,
            smoothing_size=self.smoothing_size,
            threshold=self.threshold,
            percentile=self.percentile,
            preprocess=self.preprocess
        )

        # TODO ambrosejcarr: data should always be at least pseudo-3d, this may not be necessary
        # TODO ambrosejcarr: this is where max vs. sum vs. mean would be parametrized.
        # here, total_intensity = sum, intensity = max
        new_colnames = [
            'x', 'y', 'total_intensity', 'radius', 'eccentricity', 'intensity', 'raw_mass', 'ep'
        ]
        if len(image.shape) == 3:
            attributes.columns = ['z'] + new_colnames
        else:
            attributes.columns = new_colnames

        attributes['spot_id'] = np.arange(attributes.shape[0])
        return SpotAttributes(attributes)

    def run(self, data_stack: ImageStack,
            blobs_image: Optional[Union[np.ndarray, xr.DataArray]]=None,
            reference_image_from_max_projection: bool=False) \
            -> IntensityTable:
        """
        Find spots.

        Parameters
        ----------
        data_stack : ImageStack
            Stack where we find the spots in.
        blobs_image : Union[np.ndarray, xr.DataArray]
            If provided, spots will be found in the blobs image, and intensities will be measured
            across hybs and channels. Otherwise, spots are measured independently for each channel
            and round.
        reference_image_from_max_projection : bool
            if True, compute a reference image from the maximum projection of the channels and
            z-planes

        """
        intensity_table = detect_spots(
            data_stack=data_stack,
            spot_finding_method=self.image_to_spots,
            reference_image=blobs_image,
            reference_image_from_max_projection=reference_image_from_max_projection,
            measurement_function=self.measurement_function,
            radius_is_gyration=True,
        )

        return intensity_table

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--spot-diameter", type=str, help='expected spot size')
        group_parser.add_argument(
            "--min-mass", default=4, type=int, help="minimum integrated spot intensity")
        group_parser.add_argument(
            "--max-size", default=6, type=int, help="maximum radius of gyration of brightness")
        group_parser.add_argument(
            "--separation", default=5, type=float, help="minimum distance between spots")
        group_parser.add_argument(
            "--noise-size", default=None, type=int,
            help="width of gaussian blurring kernel, in pixels")
        group_parser.add_argument(
            "--smoothing-size", default=None, type=int,
            help="odd integer. Size of boxcar (moving average) filter in pixels. Default is the "
                 "Diameter"
        )
        group_parser.add_argument(
            "--preprocess", action="store_true",
            help="if passed, gaussian and boxcar filtering are applied")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")
        group_parser.add_argument(
            "--percentile", default=None, type=float,
            help="clip bandpass below this value. Thresholding is done on already background-"
                 "subtracted images. Default 1 for integer images and 1/255 for float")
