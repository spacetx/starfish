from __future__ import division

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from showit import image
from skimage.feature import blob_log

from starfish.munge import gather
from starfish.io import Stack
from ._base import SpotFinderAlgorithmBase
from starfish.pipeline.algorithm_base import AlgorithmBase
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.pipeline.features.encoded_spots import EncodedSpots


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(self, stack, *args, **kwargs):
        self.stack = stack

        self.blobs = None
        self.num_objs = None
        self.spots_df = None
        self.spots_df_viz = None
        self.intensities = None

    @classmethod
    def from_cli_args(cls, args):
        s = Stack()
        s.read(args.input)
        return cls(s)

    @classmethod
    def get_algorithm_name(cls):
        return "gaussian_spot_detector"

    @classmethod
    def add_arguments(cls, group_parser):
        # todo better description of how aux_image gets used
        group_parser.add_argument("--blobs", type=str, help='aux image key')
        group_parser.add_argument(
            "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
        group_parser.add_argument(
            "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
        group_parser.add_argument("--num-sigma", default=20, type=int, help="Number of scales to try")
        group_parser.add_argument("--threshold", default=.01, type=float, help="Dots threshold")
        group_parser.add_argument("--show", default=False, type=bool, help="Dots threshold")

    def detect(
            self, min_sigma, max_sigma, num_sigma, threshold, blobs, measurement_type='max', bit_map_flag=False,
            *args, **kwargs):
        self.blobs = self.stack.aux_dict[blobs]
        fitted_blobs = self._fit(min_sigma, max_sigma, num_sigma, threshold)
        spots_df_viz = self._fitted_blobs_to_df(fitted_blobs)
        intensity = self._measure(self.blobs, spots_df_viz, measurement_type)
        spots_df_viz['intensity'] = intensity
        spots_df_viz['spot_id'] = spots_df_viz.index
        self.spots_df_viz = spots_df_viz  # todo this needs a class if it's going to stay around
        self.intensities = self._measure_stack(measurement_type)
        res = self._to_encoder_dataframe(bit_map_flag)
        return res

    def _fit(self, min_sigma, max_sigma, num_sigma, threshold):
        fitted_blobs = blob_log(self.blobs, min_sigma, max_sigma, num_sigma, threshold)
        fitted_blobs[:, 2] = fitted_blobs[:, 2] * np.sqrt(2)
        self.num_objs = fitted_blobs.shape[0]
        return fitted_blobs

    @staticmethod
    def _fitted_blobs_to_df(fitted_blobs):
        res = pd.DataFrame(fitted_blobs)
        res['d'] = res[2] * 2
        res.rename(columns={0: 'x', 1: 'y', 2: 'r'}, inplace=True)
        res['xmin'] = np.floor(res.x - res.r)
        res['xmax'] = np.ceil(res.x + res.r)
        res['ymin'] = np.floor(res.y - res.r)
        res['ymax'] = np.ceil(res.y + res.r)
        return res

    @staticmethod
    def _measure(img, spots_df, measurement_type):
        res = []
        for row in spots_df.iterrows():
            row = row[1]
            subset = img[int(row.xmin):int(row.xmax), int(row.ymin):int(row.ymax)]

            if measurement_type == 'max':
                res.append(subset.max())
            else:
                res.append(subset.mean())

        return res

    def _measure_stack(self, measurement_type):
        intensities = [self._measure(img, self.spots_df_viz, measurement_type) for img in self.stack.squeeze()]
        return intensities

    def _to_encoder_dataframe(self, bit_map_flag):
        self.stack.squeeze(bit_map_flag=bit_map_flag)
        mapping = self.stack.squeeze_map
        inds = range(len(self.stack.squeeze_map))
        d = dict(zip(inds, self.intensities))
        d['spot_id'] = range(self.num_objs)

        res = pd.DataFrame(d)
        res = gather(res, 'ind', 'val', inds)
        res = pd.merge(res, mapping, on='ind', how='left')

        return res

    def to_viz_dataframe(self):
        return self.spots_df_viz

    def show(self, figsize=(20, 20)):
        plt.figure(figsize=figsize)
        ax = plt.gca()

        image(self.blobs, ax=ax)
        blobs_log = self.spots_df_viz.loc[:, ['x', 'y', 'r']].values
        for blob in blobs_log:
            x, y, r = blob
            c = plt.Circle((y, x), r, color='r', linewidth=2, fill=False)
            ax.add_patch(c)

        plt.title('Num blobs: {}'.format(len(blobs_log)))
        plt.show()


class GaussianSpotDetectorNew(AlgorithmBase):

    def __init__(self, min_sigma, max_sigma, num_sigma, threshold, blobs, measurement_type='max', **kwargs):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.blobs = blobs

        try:
            self.measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                'measurement_type must be a numpy reduce function such as "max" or "mean". {} not found.'.format(
                    measurement_type)
            )

    def fit(self):
        fitted_blobs = pd.DataFrame(
            data=blob_log(self.blobs, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold),
            columns=['x', 'y', 'r']
        )
        # TODO ambrosejcarr: why is this necessary? (check docs)
        fitted_blobs['r'] *= np.sqrt(2)

        fitted_blobs['x_min'] = np.floor(fitted_blobs.x - fitted_blobs.r)
        fitted_blobs['x_max'] = np.ceil(fitted_blobs.x + fitted_blobs.r)
        fitted_blobs['y_min'] = np.floor(fitted_blobs.y - fitted_blobs.r)
        fitted_blobs['y_max'] = np.ceil(fitted_blobs.y + fitted_blobs.r)

        # TODO ambrosejcarr: are these not already ints?
        fitted_blobs['intensity'] = fitted_blobs.astype(int).apply(
            lambda row: self.measurement_function(self.blobs[row.x_min:row.x_max, row.y_min:row.y_max])
        )

        fitted_blobs['spot_id'] = np.arange(fitted_blobs.shape[0])

        return SpotAttributes(fitted_blobs)

    def run(self, image_stack) -> Tuple[SpotAttributes, EncodedSpots]:
        spot_attributes = self.fit()
        encoded_spots = spot_attributes.encode(image_stack)
        return spot_attributes, encoded_spots

    @classmethod
    def from_cli_args(cls, args):
        return cls(**vars(args))

    @classmethod
    def get_algorithm_name(cls):
        return 'gaussian_spot_detector_new'

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--blobs", type=str, help='aux image key')
        group_parser.add_argument(
            "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
        group_parser.add_argument(
            "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
        group_parser.add_argument("--num-sigma", default=20, type=int, help="Number of scales to try")
        group_parser.add_argument("--threshold", default=.01, type=float, help="Dots threshold")
        group_parser.add_argument("--show", default=False, type=bool, help="Dots threshold")


