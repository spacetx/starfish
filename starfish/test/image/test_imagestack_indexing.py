from starfish.experiment.builder import tile_fetcher_factory
from starfish.imagestack.imagestack import ImageStack
import starfish.test.image.test_imagestack_coordinates as tc


def test_indexing():
    stack = ImageStack.synthetic_stack(
        tc.NUM_ROUND, tc.NUM_CH, tc.NUM_Z,
        tc.HEIGHT, tc.WIDTH,
        tile_fetcher=tile_fetcher_factory(
            tc.OffsettedTiles,
            True,
        )
    )
    indexed = stack[1:, 0, 0]
