import os

import pandas as pd

import starfish
from starfish.core.test.factories import two_spot_sparse_coded_data_factory


def test_per_round_max_spot_decoding_without_spots(tmpdir):

    codebook, image_stack, max_intensity = two_spot_sparse_coded_data_factory()

    bd = starfish.spots.FindSpots.BlobDetector(
        min_sigma=1, max_sigma=1, num_sigma=1, threshold=max_intensity + 0.1)
    no_spots = bd.run(image_stack)

    decode = starfish.spots.DecodeSpots.PerRoundMaxChannel(codebook)
    decoded_no_spots: starfish.DecodedIntensityTable = decode.run(spots=no_spots)

    decoded_spot_table = decoded_no_spots.to_decoded_dataframe()

    filename = os.path.join(tmpdir, 'test.csv')
    decoded_spot_table.save_csv(os.path.join(tmpdir, 'test.csv'))

    # verify we can concatenate two empty tables
    table1 = pd.read_csv(filename, index_col=0)
    table2 = pd.read_csv(filename, index_col=0)
    pd.concat([table1, table2], axis=0)


def test_metric_decoding_without_spots(tmpdir):

    codebook, image_stack, max_intensity = two_spot_sparse_coded_data_factory()

    bd = starfish.spots.FindSpots.BlobDetector(
        min_sigma=1, max_sigma=1, num_sigma=1, threshold=max_intensity + 0.1)
    no_spots = bd.run(image_stack)

    decode = starfish.spots.DecodeSpots.MetricDistance(
        codebook, max_distance=0, min_intensity=max_intensity + 0.1
    )
    decoded_no_spots: starfish.DecodedIntensityTable = decode.run(spots=no_spots)

    decoded_spot_table = decoded_no_spots.to_decoded_dataframe()

    filename = os.path.join(tmpdir, 'test.csv')
    decoded_spot_table.save_csv(os.path.join(tmpdir, 'test.csv'))

    # verify we can concatenate two empty tables
    table1 = pd.read_csv(filename, index_col=0)
    table2 = pd.read_csv(filename, index_col=0)
    pd.concat([table1, table2], axis=0)
