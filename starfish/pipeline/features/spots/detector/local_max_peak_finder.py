from typing import List, Tuple

import numpy as np
import pandas as pd
from trackpy import locate

from starfish.io import Stack
from starfish.pipeline.features.encoded_spots import EncodedSpots
from starfish.pipeline.features.spot_attributes import SpotAttributes
from ._base import SpotFinderAlgorithmBase


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(
            self, spot_diameter, min_mass, max_size, separation, noise_size, smoothing_size, percentile, threshold,
            preprocess, **kwargs):
        """

        Parameters
        ----------
        spot_diameter
        min_mass
        max_size
        separation
        noise_size
        smoothing_size
        percentile
        threshold
        preprocess
        kwargs
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

    # TODO ambrosejcarr: make this generalize to barcode-containing methods (first, make munge more transparent)
    def encode(self, spot_attributes: SpotAttributes):
        spot_table = spot_attributes.data
        spot_table['barcode_index'] = np.ones(spot_table.shape[0])

    def find_attributes(self, image):
        # DataFrame([x, y, mass, size, ecc, signal])
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
        attributes.columns.names = ['x', 'y', 'z', 'intensity', 'r', 'eccentricity', 'signal']
        attributes['spot_id'] = np.arange(attributes.shape[0])
        return SpotAttributes(attributes)

    def find(self, image_stack: Stack) -> Tuple[SpotAttributes, EncodedSpots]:
        spot_attributes: List[SpotAttributes] = image_stack.image.apply(self.find_attributes, is_volume=True)

        # stick the spot_attribute tables together
        spot_attributes: SpotAttributes = pd.concat([t.data for t in spot_attributes], axis=0)

        # TODO ambrosejcarr: this is where development stopped; spot_attributes is correct, but translating
        # spot_attributes into an encoder_table is tricky without first implementing the new codebook. Do that first.
        # create an encoded table
        encoded_spots = self.encode(spot_attributes.data)

        return spot_attributes, encoded_spots

    @classmethod
    def get_algorithm_name(cls):
        return 'local_max_peak_finder'

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
            "--percentile", default=64., type=float,
            help="peaks must have brightness greater than this percentage of pixels")
        group_parser.add_argument(
            "--preprocess", action="store_true", help="if passed, gaussian and boxcar filtering are applied")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")
        group_parser.add_argument(
            "--percentile", default=None, type=float,
            help="clip bandpass below this value. Thresholding is done on already background-subtracted images. "
                 "default 1 for integer images and 1/255 for float"
        )
