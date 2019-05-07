#!/usr/bin/env python
# coding: utf-8

import numpy as np
import starfish.data
from starfish.types import Axes


def main():
    exp = starfish.data.SeqFISH(use_test_data=True)

    print("downloading data")
    img = exp['fov_000'].get_image('primary')
    img = img.sel({Axes.X: (0, 150), Axes.Y: (0, 150), Axes.ZPLANE: (0, 15)})

    print("correcting background")
    wth = starfish.image.Filter.WhiteTophat(masking_radius=3)
    background_corrected = wth.run(img, in_place=False)

    def scale_by_percentile(data, p=99.9):
        data = np.asarray(data)
        cval = np.percentile(data, p)
        data = data / cval
        data[data > 1] = 1
        return data

    print("scaling")
    scaled = background_corrected.apply(
        scale_by_percentile,
        group_by={Axes.ROUND, Axes.CH}, verbose=False, in_place=False,
        n_processes=1,
    )
    print("blurring")
    glp = starfish.image.Filter.GaussianLowPass(sigma=(0.3, 1, 1), is_volume=True)
    blurred = glp.run(scaled)

    print("decoding pixels")
    psd = starfish.spots.DetectPixels.PixelSpotDecoder(
        codebook=exp.codebook, metric='euclidean', distance_threshold=0.5,
        magnitude_threshold=0.1, min_area=7, max_area=50,
    )
    psd.run(blurred)


if __name__ == "__main__":
    main()
