import json
import os
import tempfile
import shutil

from starfish.io import Stack
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector, GaussianSpotDetectorNew
from starfish.util.test import DATA_DIR
from starfish.munge import spots_to_geojson
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.pipeline.features.encoded_spots import EncodedSpots

RESULTS_DIR = tempfile.mkdtemp()


def test_gaussian_spot_detector():

    # make a stack
    s = Stack()
    s.read(os.path.join(DATA_DIR, 'iss', 'filtered', 'experiment.json'))
    print(s.image.numpy_array.shape)

    gsd = GaussianSpotDetector(stack=s)
    spots_df_tidy = gsd.detect(
        min_sigma=4,
        max_sigma=6,
        num_sigma=20,
        threshold=0.01,
        blobs='dots',
        measurement_type='max',
        bit_map_flag=False
    )

    # mimic CLI output
    spots_viz = gsd.spots_df_viz
    geojson = spots_to_geojson(spots_viz)

    path = os.path.join(RESULTS_DIR, 'spots.geojson')
    print("Writing | spots geojson to: {}".format(path))
    with open(path, 'w') as f:
        f.write(json.dumps(geojson))

    path = os.path.join(RESULTS_DIR, 'spots.json')
    print("Writing | spot_id | x | y | z | to: {}".format(path))
    spots_viz.to_json(path, orient="records")

    path = os.path.join(RESULTS_DIR, 'encoder_table.json')
    print("Writing | spot_id | hyb | ch | val | to: {}".format(path))
    spots_df_tidy.to_json(path, orient="records")


def test_gaussian_spot_detector_new():

    s = Stack()
    s.read(os.path.join(DATA_DIR, 'iss', 'filtered', 'experiment.json'))
    print(s.image.numpy_array.shape)

    gsd = GaussianSpotDetectorNew(
        min_sigma=4,
        max_sigma=6,
        num_sigma=20,
        threshold=0.01,
        blobs='dots',
        measurement_type='max',
        bit_map_flag=False
    )
    spot_attributes, encoded_spots = gsd.run(s)

    sa2 = SpotAttributes.load_json(os.path.join(RESULTS_DIR, 'spots.json'))
    es2 = EncodedSpots.load(os.path.join(RESULTS_DIR, 'encoder_table.json'))


shutil.rmtree(RESULTS_DIR)
