import json
import os
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Tuple, Union

from slicedimage import (
    Collection,
    ImageFormat,
    Tile,
    TileSet,
    Writer,
)

from starfish.codebook.codebook import Codebook
from starfish.experiment.version import CURRENT_VERSION
from starfish.types import Coordinates, Indices
from .defaultproviders import RandomNoiseTile, tile_fetcher_factory
from .providers import FetchedTile, TileFetcher


AUX_IMAGE_NAMES = {
    'nuclei',
    'dots',
}


def _tile_opener(toc_path: str, tile: Tile, file_ext: str) -> BinaryIO:
    tile_basename = os.path.splitext(toc_path)[0]
    return open(
        "{}-Z{}-H{}-C{}.{}".format(
            tile_basename,
            tile.indices[Indices.Z],
            tile.indices[Indices.ROUND],
            tile.indices[Indices.CH],
            ImageFormat.TIFF.file_ext,
        ),
        "wb")


def _fov_path_generator(parent_toc_path: str, toc_name: str) -> str:
    toc_basename = os.path.splitext(os.path.basename(parent_toc_path))[0]
    return os.path.join(
        os.path.dirname(parent_toc_path),
        "{}-{}.json".format(toc_basename, toc_name),
    )


def build_image(
        fov_count: int, round_count: int, ch_count: int, z_count: int,
        image_fetcher: TileFetcher,
        default_shape: Optional[Tuple[int, int]]=None,
) -> Collection:
    """
    Build and returns an image set with the following characteristics:

    Parameters
    ----------
    fov_count : int
        Number of fields of view in this image set.
    round_count : int
        Number for rounds in this image set.
    ch_count : int
        Number for channels in this image set.
    z_count : int
        Number of z-layers in this image set.
    image_fetcher : TileFetcher
        Instance of TileFetcher that provides the data for the tile.
    default_shape : Optional[Tuple[int, int]]
        Default shape of the individual tiles in this image set.

    Returns
    -------
    The slicedimage collection representing the image.
    """
    collection = Collection()
    for fov_ix in range(fov_count):
        fov_images = TileSet(
            [
                Coordinates.X,
                Coordinates.Y,
                Coordinates.Z,
                Indices.Z,
                Indices.ROUND,
                Indices.CH,
                Indices.X,
                Indices.Y,
            ],
            {Indices.ROUND: round_count, Indices.CH: ch_count, Indices.Z: z_count},
            default_shape,
            ImageFormat.TIFF,
        )

        for z_ix in range(z_count):
            for round_ix in range(round_count):
                for ch_ix in range(ch_count):
                    image = image_fetcher.get_tile(fov_ix, round_ix, ch_ix, z_ix)
                    tile = Tile(
                        image.coordinates,
                        {
                            Indices.Z: z_ix,
                            Indices.ROUND: round_ix,
                            Indices.CH: ch_ix,
                        },
                        image.shape,
                        extras=image.extras,
                    )
                    tile.set_numpy_array_future(image.tile_data)
                    fov_images.add_tile(tile)
        collection.add_partition("fov_{:03}".format(fov_ix), fov_images)
    return collection


def write_experiment_json(
        path: str,
        fov_count: int,
        primary_image_dimensions: Mapping[Union[str, Indices], int],
        aux_name_to_dimensions: Mapping[str, Mapping[Union[str, Indices], int]],
        primary_tile_fetcher: Optional[TileFetcher]=None,
        aux_tile_fetcher: Optional[Mapping[str, TileFetcher]]=None,
        postprocess_func: Optional[Callable[[dict], dict]]=None,
        default_shape: Optional[Tuple[int, int]]=None,
) -> None:
    """
    Build and returns a top-level experiment description with the following characteristics:

    Parameters
    ----------
    path : str
        Directory to write the files to.
    fov_count : int
        Number of fields of view in this experiment.
    primary_image_dimensions : Mapping[Union[str, Indices], int]
        Dictionary mapping dimension name to dimension size for the primary image.
    aux_name_to_dimensions : Mapping[str, Mapping[Union[str, Indices], int]]
        Dictionary mapping the auxiliary image type to dictionaries, which map from dimension name
        to dimension size.
    primary_tile_fetcher : Optional[TileFetcher]
        TileFetcher for primary images.  Set this if you want specific image data to be set for the
        primary images.  If not provided, the image data is set to random noise via
        :class:`RandomNoiseTileFetcher`.
    aux_tile_fetcher : Optional[Mapping[str, TileFetcher]]
        TileFetchers for auxiliary images.  Set this if you want specific image data to be set for
        one or more aux image types.  If not provided for any given aux image, the image data is
        set to random noise via :class:`RandomNoiseTileFetcher`.
    postprocess_func : Optional[Callable[[dict], dict]]
        If provided, this is called with the experiment document for any postprocessing.
        An example of this would be to add something to one of the top-level extras field.
        The callable should return what is to be written as the experiment document.
    default_shape : Optional[Tuple[int, int]] (default = None)
        Default shape for the tiles in this experiment.
    """
    if primary_tile_fetcher is None:
        primary_tile_fetcher = tile_fetcher_factory(RandomNoiseTile)
    if aux_tile_fetcher is None:
        aux_tile_fetcher = {}
    if postprocess_func is None:
        postprocess_func = lambda doc: doc

    experiment_doc: Dict[str, Any] = {
        'version': str(CURRENT_VERSION),
        'images': {},
        'extras': {},
    }
    primary_image = build_image(
        fov_count,
        primary_image_dimensions[Indices.ROUND],
        primary_image_dimensions[Indices.CH],
        primary_image_dimensions[Indices.Z],
        primary_tile_fetcher,
        default_shape=default_shape,
    )
    Writer.write_to_path(
        primary_image,
        os.path.join(path, "primary_image.json"),
        pretty=True,
        partition_path_generator=_fov_path_generator,
        tile_opener=_tile_opener,
        tile_format=ImageFormat.TIFF,
    )
    experiment_doc['images']['primary'] = "primary_image.json"

    for aux_name, aux_dimensions in aux_name_to_dimensions.items():
        if aux_dimensions is None:
            continue
        auxiliary_image = build_image(
            fov_count,
            aux_dimensions[Indices.ROUND], aux_dimensions[Indices.CH], aux_dimensions[Indices.Z],
            aux_tile_fetcher.get(aux_name, tile_fetcher_factory(RandomNoiseTile)),
            default_shape=default_shape,
        )
        Writer.write_to_path(
            auxiliary_image,
            os.path.join(path, "{}.json".format(aux_name)),
            pretty=True,
            partition_path_generator=_fov_path_generator,
            tile_opener=_tile_opener,
            tile_format=ImageFormat.TIFF,
        )
        experiment_doc['images'][aux_name] = "{}.json".format(aux_name)

    experiment_doc = postprocess_func(experiment_doc)
    experiment_doc["codebook"] = "codebook.json"
    with open(os.path.join(path, "experiment.json"), "w") as fh:
        json.dump(experiment_doc, fh, indent=4)

    codebook_array = [
        {
            "codeword": [
                {"r": 0, "c": 0, "v": 1},
            ],
            "target": "PLEASE_REPLACE_ME"
        },
    ]
    codebook = Codebook.from_code_array(codebook_array)
    codebook_json_filename = "codebook.json"
    codebook.to_json(os.path.join(path, codebook_json_filename))
