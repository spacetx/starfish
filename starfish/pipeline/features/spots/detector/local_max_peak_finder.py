from typing import List

import numpy as np
from trackpy import locate

from starfish.image import ImageStack
from starfish.pipeline.features.spot_attributes import SpotAttributes
from ._base import SpotFinderAlgorithmBase


class LocalMaxPeakFinder(SpotFinderAlgorithmBase):

    def __init__(
            self, spot_diameter, min_mass, max_size, separation, percentile=0, noise_size=None, smoothing_size=None,
            threshold=None, preprocess: bool=False, is_volume: bool=False, verbose=False, **kwargs
    ) -> None:
        """Local max peak finding algorithm

        This is a wrapper for `trackpy.locate`

        Parameters
        ----------

        spot_diameter : odd integer or tuple of odd integers.
            This may be a single number or a tuple giving the feature’s extent in each dimension, useful when the
            dimensions do not have equal resolution (e.g. confocal microscopy). The tuple order is the same as the
            image shape, conventionally (z, y, x) or (y, x). The number(s) must be odd integers. When in doubt, round
            up.
        min_mass : float, optional
            The minimum integrated brightness. This is a crucial parameter for eliminating spurious features.
            Recommended minimum values are 100 for integer images and 1 for float images. Defaults to 0 (no filtering).
        max_size : float
            maximum radius-of-gyration of brightness, default None
        separation : float or tuple
            Minimum separtion between features. Default is diameter + 1. May be a tuple, see diameter for details.
        percentile : float
            Features must have a peak brighter than pixels in this percentile. This helps eliminate spurious peaks.
        noise_size : float or tuple
            Width of Gaussian blurring kernel, in pixels Default is 1. May be a tuple, see diameter for details.
        smoothing_size : float or tuple
            The size of the sides of the square kernel used in boxcar (rolling average) smoothing, in pixels Default
            is diameter. May be a tuple, making the kernel rectangular.
        threshold : float
            Clip bandpass result below this value. Thresholding is done on the already background-subtracted image.
            By default, 1 for integer images and 1/255 for float images.
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
        self.preprocess = preprocess
        self.is_volume = is_volume
        self.verbose = verbose

    # # TODO ambrosejcarr: make this generalize to smFISH methods
    # def encode(self, spot_attributes: SpotAttributes):
    #     spot_table = spot_attributes.data
    #     spot_table['barcode_index'] = np.ones(spot_table.shape[0])

    def find_attributes(self, image: np.ndarray) -> SpotAttributes:
        """

        Parameters
        ----------
        image : np.ndarray
            two- or three-dimensional numpy array containing spots to detect

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

        new_colnames = ['x', 'y', 'intensity', 'r', 'eccentricity', 'signal', 'raw_mass', 'ep']
        if len(image.shape) == 3:
            attributes.columns = ['z'] + new_colnames
        else:
            attributes.columns = new_colnames

        attributes['spot_id'] = np.arange(attributes.shape[0])
        return SpotAttributes(attributes)

    def find(self, stack: ImageStack):
        """
        Find spots.

        Parameters
        ----------
        stack : ImageStack
            Stack where we find the spots in.
        """
        spot_attributes: List[SpotAttributes] = stack.transform(
            self.find_attributes, is_volume=self.is_volume, verbose=self.verbose)

        # TODO ambrosejcarr: do we need to find spots in the aux_dict too?

        # TODO ambrosejcarr: this is where development stopped; spot_attributes is correct, but translating
        # spot_attributes into an encoder_table is tricky without first implementing the new codebook. Do that first.
        # create an encoded table
        # encoded_spots = self.encode(spot_attributes.data)

        return spot_attributes

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--spot-diameter", type=str, help='expected spot size')
        group_parser.add_argument("--min-mass", default=4, type=int, help="minimum integrated spot intensity")
        group_parser.add_argument("--max-size", default=6, type=int, help="maximum radius of gyration of brightness")
        group_parser.add_argument("--separation", default=5, type=float, help="minimum distance between spots")
        group_parser.add_argument(
            "--noise-size", default=None, type=int, help="width of gaussian blurring kernel, in pixels")
        group_parser.add_argument(
            "--smoothing-size", default=None, type=int,
            help="odd integer. Size of boxcar (moving average) filter in pixels. Default is the Diameter")
        group_parser.add_argument(
            "--preprocess", action="store_true", help="if passed, gaussian and boxcar filtering are applied")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")
        group_parser.add_argument(
            "--percentile", default=None, type=float,
            help="clip bandpass below this value. Thresholding is done on already background-subtracted images. "
                 "default 1 for integer images and 1/255 for float"
        )
