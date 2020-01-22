from starfish.core.experiment.builder.builder import build_image
from starfish.core.experiment.builder.defaultproviders import OnesTile, tile_fetcher_factory
from starfish.core.experiment.builder.providers import TileFetcher
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes


def synthetic_stack(
        num_round: int = 4,
        num_ch: int = 4,
        num_z: int = 12,
        tile_height: int = 50,
        tile_width: int = 40,
        tile_fetcher: TileFetcher = None,
) -> ImageStack:
    """generate a synthetic ImageStack

    Returns
    -------
    ImageStack :
        imagestack containing a tensor whose default shape is (2, 3, 4, 30, 20)
        and whose default values are all 1.

    """
    if tile_fetcher is None:
        tile_fetcher = tile_fetcher_factory(
            OnesTile,
            False,
            {Axes.Y: tile_height, Axes.X: tile_width},
        )

    collection = build_image(
        range(1),
        range(num_round),
        range(num_ch),
        range(num_z),
        tile_fetcher,
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset)
