import json
import os
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from slicedimage import (
    Collection,
    ImageFormat,
    Tile,
    TileSet,
    Writer,
)

from starfish.core.codebook.codebook import Codebook
from starfish.core.experiment.builder.orderediterator import join_axes_labels, ordered_iterator
from starfish.core.experiment.version import CURRENT_VERSION
from starfish.core.types import Axes, Coordinates
from .defaultproviders import RandomNoiseTile, tile_fetcher_factory
from .providers import FetchedTile, TileFetcher


AUX_IMAGE_NAMES = {
    'nuclei',
    'dots',
}
DEFAULT_DIMENSION_ORDER = (Axes.ZPLANE, Axes.ROUND, Axes.CH)


def _tile_opener(toc_path: Path, tile: Tile, file_ext: str) -> BinaryIO:
    base = toc_path.parent / toc_path.stem
    return open(
        "{}-Z{}-H{}-C{}.{}".format(
            str(base),
            tile.indices[Axes.ZPLANE],
            tile.indices[Axes.ROUND],
            tile.indices[Axes.CH],
            ImageFormat.TIFF.file_ext,
        ),
        "wb")


def _fov_path_generator(parent_toc_path: Path, toc_name: str) -> Path:
    return parent_toc_path.parent / "{}-{}.json".format(parent_toc_path.stem, toc_name)


def build_image(
        fovs: Sequence[int],
        rounds: Sequence[int],
        chs: Sequence[int],
        zplanes: Sequence[int],
        image_fetcher: TileFetcher,
        default_shape: Optional[Mapping[Axes, int]]=None,
        axes_order: Sequence[Axes]=DEFAULT_DIMENSION_ORDER,
) -> Collection:
    """
    Build and returns an image set with the following characteristics:

    Parameters
    ----------
    fovs : Sequence[int]
        Sequence of field of view ids in this image set.
    rounds : Sequence[int]
        Sequence of the round numbers in this image set.
    chs : Sequence[int]
        Sequence of the ch numbers in this image set.
    zplanes : Sequence[int]
        Sequence of the zplane numbers in this image set.
    image_fetcher : TileFetcher
        Instance of TileFetcher that provides the data for the tile.
    default_shape : Optional[Tuple[int, int]]
        Default shape of the individual tiles in this image set.
    axes_order : Sequence[Axes]
        Ordering for which axes vary, in order of the slowest changing axis to the fastest.  For
        instance, if the order is (ROUND, Z, CH) and each dimension has size 2, then the sequence
        is:
          (ROUND=0, CH=0, Z=0)
          (ROUND=0, CH=1, Z=0)
          (ROUND=0, CH=0, Z=1)
          (ROUND=0, CH=1, Z=1)
          (ROUND=1, CH=0, Z=0)
          (ROUND=1, CH=1, Z=0)
          (ROUND=1, CH=0, Z=1)
          (ROUND=1, CH=1, Z=1)
        (default = (Axes.Z, Axes.ROUND, Axes.CH))

    Returns
    -------
    The slicedimage collection representing the image.
    """
    axes_sizes = join_axes_labels(
        axes_order, rounds=rounds, chs=chs, zplanes=zplanes)

    collection = Collection()
    for fov_id in fovs:
        fov_images = TileSet(
            [
                Coordinates.X,
                Coordinates.Y,
                Coordinates.Z,
                Axes.ZPLANE,
                Axes.ROUND,
                Axes.CH,
                Axes.X,
                Axes.Y,
            ],
            {Axes.ROUND: len(rounds), Axes.CH: len(chs), Axes.ZPLANE: len(zplanes)},
            default_shape,
            ImageFormat.TIFF,
        )

        for selector in ordered_iterator(axes_sizes):
            image = image_fetcher.get_tile(
                fov_id,
                selector[Axes.ROUND],
                selector[Axes.CH],
                selector[Axes.ZPLANE])
            tile = Tile(
                image.coordinates,
                {
                    Axes.ZPLANE: (selector[Axes.ZPLANE]),
                    Axes.ROUND: (selector[Axes.ROUND]),
                    Axes.CH: (selector[Axes.CH]),
                },
                image.shape,
                extras=image.extras,
            )
            tile.set_numpy_array_future(image.tile_data)
            # Astute readers might wonder why we set this variable.  This is to support in-place
            # experiment construction.  We monkey-patch slicedimage's Tile class such that checksum
            # computation is done by finding the FetchedTile object, which allows us to calculate
            # the checksum of the original file.
            tile.provider = image
            fov_images.add_tile(tile)
        collection.add_partition("fov_{:03}".format(fov_id), fov_images)
    return collection


def write_experiment_json(
        path: str,
        fov_count: int,
        tile_format: ImageFormat,
        *,
        primary_image_dimensions: Mapping[Union[str, Axes], int],
        aux_name_to_dimensions: Mapping[str, Mapping[Union[str, Axes], int]],
        primary_tile_fetcher: Optional[TileFetcher]=None,
        aux_tile_fetcher: Optional[Mapping[str, TileFetcher]]=None,
        postprocess_func: Optional[Callable[[dict], dict]]=None,
        default_shape: Optional[Mapping[Axes, int]]=None,
        dimension_order: Sequence[Axes]=(Axes.ZPLANE, Axes.ROUND, Axes.CH),
        fov_path_generator: Callable[[Path, str], Path] = _fov_path_generator,
        tile_opener: Callable[[Path, Tile, str], BinaryIO] = _tile_opener,
) -> None:
    """
    Build and returns a top-level experiment description with the following characteristics:

    Parameters
    ----------
    path : str
        Directory to write the files to.
    fov_count : int
        Number of fields of view in this experiment.
    tile_format : ImageFormat
        File format to write the tiles as.
    primary_image_dimensions : Mapping[Union[str, Axes], int]
        Dictionary mapping dimension name to dimension size for the primary image.
    aux_name_to_dimensions : Mapping[str, Mapping[Union[str, Axes], int]]
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
    dimension_order : Sequence[Axes]
        Ordering for which dimensions vary, in order of the slowest changing dimension to the
        fastest.  For instance, if the order is (ROUND, Z, CH) and each dimension has size 2, then
        the sequence is:
          (ROUND=0, CH=0, Z=0)
          (ROUND=0, CH=1, Z=0)
          (ROUND=0, CH=0, Z=1)
          (ROUND=0, CH=1, Z=1)
          (ROUND=1, CH=0, Z=0)
          (ROUND=1, CH=1, Z=0)
          (ROUND=1, CH=0, Z=1)
          (ROUND=1, CH=1, Z=1)
        (default = (Axes.Z, Axes.ROUND, Axes.CH))
    fov_path_generator : Callable[[Path, str], Path]
        Generates the path for a FOV's json file.  If one is not provided, the default generates
        the FOV's json file at the same level as the top-level json file for an image.
    tile_opener : Callable[[Path, Tile, str], BinaryIO]
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
        range(fov_count),
        range(primary_image_dimensions[Axes.ROUND]),
        range(primary_image_dimensions[Axes.CH]),
        range(primary_image_dimensions[Axes.ZPLANE]),
        primary_tile_fetcher,
        axes_order=dimension_order,
        default_shape=default_shape,
    )
    Writer.write_to_path(
        primary_image,
        os.path.join(path, "primary_images.json"),
        pretty=True,
        partition_path_generator=fov_path_generator,
        tile_opener=tile_opener,
        tile_format=tile_format,
    )
    experiment_doc['images']['primary'] = "primary_images.json"

    for aux_name, aux_dimensions in aux_name_to_dimensions.items():
        if aux_dimensions is None:
            continue
        auxiliary_image = build_image(
            range(fov_count),
            range(aux_dimensions[Axes.ROUND]),
            range(aux_dimensions[Axes.CH]),
            range(aux_dimensions[Axes.ZPLANE]),
            aux_tile_fetcher.get(aux_name, tile_fetcher_factory(RandomNoiseTile)),
            axes_order=dimension_order,
            default_shape=default_shape,
        )
        Writer.write_to_path(
            auxiliary_image,
            os.path.join(path, "{}.json".format(aux_name)),
            pretty=True,
            partition_path_generator=fov_path_generator,
            tile_opener=tile_opener,
            tile_format=tile_format,
        )
        experiment_doc['images'][aux_name] = "{}.json".format(aux_name)

    experiment_doc["codebook"] = "codebook.json"
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

    experiment_doc = postprocess_func(experiment_doc)

    with open(os.path.join(path, "experiment.json"), "w") as fh:
        json.dump(experiment_doc, fh, indent=4)
