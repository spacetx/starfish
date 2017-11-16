import numpy as np
from skimage.measure import regionprops, label
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class MerfishDecoder:
    def __init__(self, encoded, codebook):
        self.encoded = encoded
        self.codebook = codebook

        self.decoded_df = None
        self.decoded_img = None
        self.label_image = None
        self.spot_props = None

    def decode(self,
               img_size=(2048, 2048),
               distance_threshold=0.5176,
               magnitude_threshold=1,
               area_threshold=2,
               crop_size=40):

        codes = self._parse_barcodes()
        pixel_traces, pixel_traces_l2_norm = self._parse_pixel_traces()

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(codes)
        distances, indices = nn.kneighbors(pixel_traces)

        # revert back to image space
        decoded_img = np.reshape(indices + 1, img_size)
        decoded_dist = np.reshape(distances, img_size)
        local_magnitude = np.reshape(pixel_traces_l2_norm, img_size)

        # find good 'spots'. filter out bad 'spots'
        decoded_img[decoded_dist > distance_threshold] = 0
        decoded_img[local_magnitude < magnitude_threshold] = 0
        decoded_img = self._crop(decoded_img, crop_size)
        self.decoded_img = decoded_img

        decoded_df = self._find_spots(decoded_img, area_threshold)
        decoded_df = pd.merge(decoded_df, self.codebook, on='barcode', how='left')
        self.decoded_df = decoded_df
        return self.decoded_df

    def _parse_barcodes(self):
        # parse barcode into numpy array and normalize by l2_norm
        codes = np.array([np.array([int(d) for d in c]) for c in self.codebook.barcode])
        codes_l2_norm = np.linalg.norm(codes, axis=1, ord=2)
        weighted_codes = codes / codes_l2_norm[:, None]
        return weighted_codes

    def _parse_pixel_traces(self):
        # parse spots into pixel traces, normalize and filter
        df = self.encoded.loc[:, ['spot_id', 'bit', 'val']]
        # TODO this assumes that bits are sorted
        pixel_traces = df.pivot(index='spot_id', columns='bit', values='val')
        pixel_traces = pixel_traces.values
        pixel_traces_l2_norm = np.linalg.norm(pixel_traces, axis=1, ord=2)
        ind = pixel_traces_l2_norm > 0
        pixel_traces[ind, :] = pixel_traces[ind, :] / pixel_traces_l2_norm[ind, None]
        return pixel_traces, pixel_traces_l2_norm

    def _crop(self, decoded_img, crop_size):
        decoded_img[:, 0:crop_size] = 0
        decoded_img[:, decoded_img.shape[1] - crop_size:] = 0
        decoded_img[0:crop_size, :] = 0
        decoded_img[decoded_img.shape[0] - crop_size:, :] = 0
        return decoded_img

    def _find_spots(self, decoded_img, area_threshold):
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

        self.label_image = label_image
        self.spot_props = props

        return spots_df
