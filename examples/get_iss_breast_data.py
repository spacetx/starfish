import argparse
import io
import os
from typing import IO, Tuple

from skimage.io import imread, imsave
from slicedimage import ImageFormat

from starfish.experiment.builder import FetchedImage, ImageFetcher, write_experiment_json
from starfish.types import Indices
from starfish.util.argparse import FsExistsType

SHAPE = (1044, 1390)


class CroppedISSImage(FetchedImage):
    def __init__(self, file_path):
        self.file_path = file_path

    @property
    def shape(self) -> Tuple[int, ...]:
        # TODO these images need to be hella cropped
        return SHAPE

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.TIFF

    @staticmethod
    def crop(img):
        crp = img[40:1084, 20:1410]
        return crp

    def image_data_handle(self) -> IO:
        im = self.crop(imread(self.file_path))
        fh = io.BytesIO()
        imsave(fh, im, plugin='tifffile')
        fh.seek(0)
        return fh


class HybridizationImageFetcher(ImageFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    @property
    def ch_dict(self):
        ch_str = ['Cy3 5', 'Cy3', 'Cy5', 'FITC']
        ch = [2, 1, 3, 0]
        ch_dict = dict(zip(ch, ch_str))
        return ch_dict

    @property
    def hyb_dict(self):
        hyb_str = ['1st', '2nd', '3rd', '4th']
        return dict(zip(range(4), hyb_str))

    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        filename = 'slideA_' + str(fov + 1) + '_' + \
                   self.hyb_dict[hyb] + '_' + self.ch_dict[ch] + '.TIF'
        file_path = os.path.join(self.input_dir, filename)
        return CroppedISSImage(file_path)


class AuxImageFetcherDAPI(ImageFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        filename = 'slideA_' + str(fov + 1) + '_DO_' + 'DAPI.TIF'
        file_path = os.path.join(self.input_dir, filename)
        return CroppedISSImage(file_path)


class AuxImageFetcherDots(ImageFetcher):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def get_image(self, fov: int, hyb: int, ch: int, z: int) -> FetchedImage:
        filename = 'slideA_' + str(fov + 1) + '_DO_' + 'Cy3.TIF'
        file_path = os.path.join(self.input_dir, filename)
        return CroppedISSImage(file_path)


def format_data(input_dir, output_dir):
    num_fovs = 16

    hyb_dimensions = {
        Indices.ROUND: 4,
        Indices.CH: 4,
        Indices.Z: 1,
    }

    aux_name_to_dimensions = {
        'nuclei': {
            Indices.ROUND: 1,
            Indices.CH: 1,
            Indices.Z: 1,
        },
        'dots': {
            Indices.ROUND: 1,
            Indices.CH: 1,
            Indices.Z: 1,
        }
    }

    hfetch = HybridizationImageFetcher(input_dir)

    auxfetch = {
        'nuclei': AuxImageFetcherDAPI(input_dir),
        'dots': AuxImageFetcherDots(input_dir),
    }

    write_experiment_json(output_dir,
                          num_fovs,
                          hyb_dimensions,
                          aux_name_to_dimensions,
                          hyb_image_fetcher=hfetch,
                          aux_image_fetcher=auxfetch,
                          default_shape=SHAPE
                          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir)
