import collections
import math
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
from skimage.measure._regionprops import _RegionProperties
from sklearn.neighbors import NearestNeighbors

from starfish.munge import melt
from starfish.io import Stack
from ._base import PixelFinderAlgorithmBase


class PixelSpotDetector(PixelFinderAlgorithmBase):

    _MerfishDecoderResults = collections.namedtuple(
        '_MerfishDecoderResults',
        ['result', 'decoded_img', 'label_img', 'spot_props'])

    def __init__(
            self, codebook: str, distance_threshold: float=0.5176,
            magnitude_threshold: int=1, area_threshold: int=2, crop_size: int=40, **kwargs) -> None:
        """Decode an image by first coding each pixel, then combining the results into spots

        Parameters
        ----------
        codebook : str
            file mapping codewords to the genes they represent
        distance_threshold : float
            spots whose codewords are more than this distance from an expected code are filtered (default  0.5176)
        magnitude_threshold : int
            spots with intensity less than this value are filtered (default 1)
        area_threshold : int
            spots with total area less than this value are filtered
        crop_size : int
            ??? # TODO ambrosejcarr what is this?

        """
        self.codebook = pd.read_csv(codebook, dtype={'barcode': object})
        self.weighted_codes: pd.DataFrame = self._normalize_barcodes(self.codebook)
        self.distance_threshold = distance_threshold
        self.magnitude_threshold = magnitude_threshold
        self.area_threshold = area_threshold
        self.crop_size = crop_size

        self.label_image: Optional[np.ndarray] = None
        self.decoded_image: Optional[np.ndarray] = None

    @staticmethod
    def encode(stack: Stack) -> pd.DataFrame:

        sq = stack.image.squeeze()
        tile_metadata = stack.image.tile_metadata
        num_bits = int(tile_metadata['barcode_index'].max() + 1)

        # linearize the pixels, mat.shape = (n_hybs * n_channels * n_z_slice, x * y)
        mat = np.reshape(sq.copy(), (sq.shape[0], sq.shape[1] * sq.shape[2]))

        res = pd.DataFrame(mat.T)
        res['spot_id'] = range(len(res))
        res = melt(
            df=res,
            new_index_name='barcode_index',
            new_value_name='intensity',
            melt_columns=range(num_bits)
        )
        # TODO this will be missing spot attributes 'r', because that comes later..., so this can't be attributes, yet
        spots_df_tidy = pd.merge(res, tile_metadata, on='barcode_index', how='left')

        return spots_df_tidy

    def find(self, stack: Stack) -> Tuple[pd.DataFrame, _MerfishDecoderResults]:
        encoded_df: pd.DataFrame = self.encode(stack)
        img_size: Tuple[int, int] = stack.image.tile_shape
        decoded_results = self.decode(encoded=encoded_df, img_size=img_size)

        return encoded_df, decoded_results

    def decode(self, encoded: pd.DataFrame, img_size: Tuple[int, int]) -> _MerfishDecoderResults:
        pixel_traces, pixel_traces_l2_norm = self._parse_pixel_traces(encoded)

        # TODO ambrosejcarr: clean up code in this function
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.weighted_codes)
        distances, indices = nn.kneighbors(pixel_traces)

        # revert back to image space
        decoded_img = np.reshape(indices + 1, img_size)
        decoded_dist = np.reshape(distances, img_size)
        local_magnitude = np.reshape(pixel_traces_l2_norm, img_size)

        # find good 'spots'. filter out bad 'spots'
        decoded_img[decoded_dist > self.distance_threshold] = 0
        decoded_img[local_magnitude < self.magnitude_threshold] = 0
        decoded_img = self._crop(decoded_img, self.crop_size)

        spot_props, label_image, decoded_df = self._find_spots(decoded_img, self.area_threshold)
        decoded_df = pd.merge(decoded_df, self.codebook, on='barcode', how='left')

        # calculate radius of each spot, assuming area is circular
        encoded['radius'] = np.sqrt(decoded_df['area'] / math.pi)
        # encoded[['x, y']] = decoded_df[['x, y']]
        # decoded_df = decoded_df.drop(['x', 'y'], axis=1)
        # spot_attributes = SpotAttributes(encoded)

        self.decoded_image = decoded_img
        self.label_image = label_image

        # DecodedSpots(decoded_df)

        return self._MerfishDecoderResults(decoded_df, decoded_img, label_image, spot_props)

    @staticmethod
    def _normalize_barcodes(codebook: pd.DataFrame) -> np.ndarray:
        # parse barcode into numpy array and normalize by l2_norm
        codes = np.array([np.array([int(d) for d in c]) for c in codebook.barcode])
        codes_l2_norm = np.linalg.norm(codes, axis=1, ord=2)
        weighted_codes = codes / codes_l2_norm[:, None]
        return weighted_codes

    # TODO ambrosejcarr fill in these values + add docs
    @staticmethod
    def _parse_pixel_traces(encoded: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # parse spots into pixel traces, normalize and filter
        df = encoded.loc[:, ['spot_id', 'barcode_index', 'intensity']]
        # TODO this assumes that bits are sorted [they are, currently]
        pixel_traces = df.pivot(index='spot_id', columns='barcode_index', values='intensity')
        pixel_traces = pixel_traces.values
        pixel_traces_l2_norm = np.linalg.norm(pixel_traces, axis=1, ord=2)
        ind = pixel_traces_l2_norm > 0
        pixel_traces[ind, :] = pixel_traces[ind, :] / pixel_traces_l2_norm[ind, None]
        return pixel_traces, pixel_traces_l2_norm

    @staticmethod
    def _crop(decoded_img, crop_size) -> np.ndarray:
        decoded_img[:, 0:crop_size] = 0
        decoded_img[:, decoded_img.shape[1] - crop_size:] = 0
        decoded_img[0:crop_size, :] = 0
        decoded_img[decoded_img.shape[0] - crop_size:, :] = 0
        return decoded_img

    def _find_spots(self, decoded_img, area_threshold) -> Tuple[List[_RegionProperties], np.ndarray, pd.DataFrame]:
        label_image = label(decoded_img, connectivity=2)
        props = regionprops(label_image)

        spots = []
        for r in props:
            if r.area >= area_threshold:
                index = decoded_img[int(r.centroid[0]), int(r.centroid[1])]
                if index > 0:
                    data = {'barcode': self.codebook.barcode[index - 1],
                            'x': r.centroid[0],
                            'y': r.centroid[1],
                            'area': r.area
                            }
                    spots.append(data)

        spots_df = pd.DataFrame(spots)
        spots_df['spot_id'] = range(len(spots_df))

        return props, label_image, spots_df

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--codebook", help="csv file containing a codebook")
        group_parser.add_argument(
            "--distance-threshold", default=0.5176,
            help="maximum distance a pixel may be from a codeword before it is filtered")
        group_parser.add_argument("--magnitude-threshold", type=float, default=1, help="minimum magnitude of a feature")
        group_parser.add_argument("--area-threshold", type=float, default=2, help="minimum area of a feature")
        # TODO ambrosejcarr: figure out help.
        group_parser.add_argument("--crop-size", type=int, default=40, help="???")
