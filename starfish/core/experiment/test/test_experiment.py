from typing import Tuple

import numpy as np
from slicedimage import Tile, TileSet

import starfish.data
from starfish.core.test.factories import SyntheticData
from starfish.types import Axes, Coordinates
from ..experiment import Experiment, FieldOfView


def round_to_x(r: int) -> Tuple[float, float]:
    return (r + 1) * 1000, (r + 1) * 100


def round_to_y(r: int) -> Tuple[float, float]:
    return (r + 1) * 10, (r + 1) * 0.1


def round_to_z(r: int) -> Tuple[float, float]:
    return (r + 1) * 0.01, (r + 1) * 0.001


NUM_ROUND = 5
NUM_CH = 2
NUM_Z = 1
HEIGHT = 100
WIDTH = 100


def get_aligned_tileset():
    alignedTileset = TileSet(
        [Axes.X, Axes.Y, Axes.CH, Axes.ZPLANE, Axes.ROUND],
        {Axes.CH: NUM_CH, Axes.ROUND: NUM_ROUND, Axes.ZPLANE: NUM_Z},
        {Axes.Y: HEIGHT, Axes.X: WIDTH})

    for r in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                tile = Tile(
                    {
                        Coordinates.X: 1,
                        Coordinates.Y: 4,
                        Coordinates.Z: 3,
                    },
                    {
                        Axes.ROUND: r,
                        Axes.CH: ch,
                        Axes.ZPLANE: z,
                    }
                )
                tile.numpy_array = np.zeros((100, 100))
                alignedTileset.add_tile(tile)
    return alignedTileset


def get_un_aligned_tileset():
    unAlignedTileset = TileSet(
        [Axes.X, Axes.Y, Axes.CH, Axes.ZPLANE, Axes.ROUND],
        {Axes.CH: NUM_CH, Axes.ROUND: NUM_ROUND, Axes.ZPLANE: NUM_Z},
        {Axes.Y: HEIGHT, Axes.X: WIDTH})

    for r in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                tile = Tile(
                    {
                        # The round_to methods generate coordinates
                        # based on the r value, therefore the coords vary
                        # throughout the tileset
                        Coordinates.X: round_to_x(r),
                        Coordinates.Y: round_to_y(r),
                        Coordinates.Z: round_to_z(r),
                    },
                    {
                        Axes.ROUND: r,
                        Axes.CH: ch,
                        Axes.ZPLANE: z,
                    }
                )
                tile.numpy_array = np.zeros((HEIGHT, WIDTH))
                unAlignedTileset.add_tile(tile)
    return unAlignedTileset


def test_fov_order():
    data = SyntheticData()
    codebook = data.codebook()
    tilesets = {"primary": get_aligned_tileset()}
    fovs = [FieldOfView("stack2", tilesets),
            FieldOfView("stack1", tilesets)]
    extras = {"synthetic": True}
    experiment = Experiment(fovs, codebook, extras)
    assert "stack1" == experiment.fov().name
    assert ["stack1", "stack2"] == [x.name for x in experiment.fovs()]


def test_crop_experiment():
    exp = starfish.data.ISS(use_test_data=True)
    image = exp['fov_001'].get_image('primary', x=slice(10, 30), y=slice(40, 70))
    assert image.shape['x'] == 20
    assert image.shape['y'] == 30

    image = exp['fov_001'].get_image('primary', rounds=[0, 1], chs=[2, 3])
    assert image.num_rounds == 2
    assert image.axis_labels(Axes.ROUND) == [0, 1]
    assert image.num_chs == 2
    assert image.axis_labels(Axes.CH) == [2, 3]


def test_fov_aligned_tileset():
    tilesets = {'primary': get_aligned_tileset(), 'nuclei': get_aligned_tileset()}
    fov = FieldOfView("aligned", tilesets)
    primary_images = fov.get_images('primary')
    nuclei_images = fov.get_images('nuclei')
    # Assert that only one ImageStack in list
    assert len(primary_images) == 1
    assert len(nuclei_images) == 1


def test_fov_un_aligned_tileset():
    tilesets = {'primary': get_un_aligned_tileset(), 'nuclei': get_un_aligned_tileset()}
    fov = FieldOfView("unaligned", tilesets)
    primary_images = fov.get_images('primary')
    nuclei_images = fov.get_images('nuclei')
    # Assert that the number of coordinate groups == NUM_ROUNDS
    assert len(primary_images) == NUM_ROUND
    assert len(nuclei_images) == NUM_ROUND
