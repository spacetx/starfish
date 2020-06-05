import functools
import json
import os
import warnings
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
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
    WriterContract,
)

from starfish.core.codebook.codebook import Codebook
from starfish.core.errors import DataFormatWarning
from starfish.core.experiment.builder.orderediterator import join_axes_labels, ordered_iterator
from starfish.core.experiment.experiment import FieldOfView
from starfish.core.experiment.version import CURRENT_VERSION
from starfish.core.types import Axes, Coordinates
from .defaultproviders import RandomNoiseTile, tile_fetcher_factory
from .providers import TileFetcher


DEFAULT_DIMENSION_ORDER: Sequence[Axes] = (Axes.ZPLANE, Axes.ROUND, Axes.CH)


@dataclass(eq=True, order=True, frozen=True)
class TileIdentifier:
    """Data class for encapsulating the location of a tile in a 6D tensor (fov, round, ch, zplane,
    y, and x)."""
    fov_id: int
    round_label: int
    ch_label: int
    zplane_label: int


def build_irregular_image(
        tile_identifiers: Iterable[TileIdentifier],
        image_fetcher: TileFetcher,
        default_shape: Optional[Mapping[Axes, int]] = None,
) -> Collection:
    """
    Build and returns an image set that can potentially be irregular (i.e., the cardinality of the
    dimensions are not always consistent).  It can also build a regular image.

    Parameters
    ----------
    tile_identifiers : Iterable[TileIdentifier]
        Iterable of all the TileIdentifier that are valid in the image.
    image_fetcher : TileFetcher
        Instance of TileFetcher that provides the data for the tile.
    default_shape : Optional[Tuple[int, int]]
        Default shape of the individual tiles in this image set.

    Returns
    -------
    The slicedimage collection representing the image.
    """
    def reducer_to_sets(
            accumulated: Sequence[MutableSet[int]], update: TileIdentifier,
    ) -> Sequence[MutableSet[int]]:
        """Reduces to a list of sets of tile identifiers, in the order of FOV, round, ch, and
        zplane."""
        result: MutableSequence[MutableSet[int]] = list()
        for accumulated_elem, update_elem in zip(accumulated, astuple(update)):
            accumulated_elem.add(update_elem)
            result.append(accumulated_elem)
        return result
    initial_value: Sequence[MutableSet[int]] = tuple(set() for _ in range(4))

    fovs, rounds, chs, zplanes = functools.reduce(
        reducer_to_sets, tile_identifiers, initial_value)

    collection = Collection()
    for expected_fov in fovs:
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

        for tile_identifier in tile_identifiers:
            current_fov, current_round, current_ch, current_zplane = astuple(tile_identifier)
            # filter out the fovs that are not the one we are currently processing
            if expected_fov != current_fov:
                continue
            image = image_fetcher.get_tile(
                current_fov,
                current_round,
                current_ch,
                current_zplane
            )
            for axis in (Axes.X, Axes.Y):
                if image.shape[axis] < max(len(rounds), len(chs), len(zplanes)):
                    warnings.warn(
                        f"{axis} axis appears to be smaller than rounds/chs/zplanes, which is "
                        "unusual",
                        DataFormatWarning
                    )

            tile = Tile(
                image.coordinates,
                {
                    Axes.ZPLANE: current_zplane,
                    Axes.ROUND: current_round,
                    Axes.CH: current_ch,
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
        collection.add_partition("fov_{:03}".format(expected_fov), fov_images)
    return collection


def build_image(
        fovs: Sequence[int],
        rounds: Sequence[int],
        chs: Sequence[int],
        zplanes: Sequence[int],
        image_fetcher: TileFetcher,
        default_shape: Optional[Mapping[Axes, int]] = None,
        axes_order: Sequence[Axes] = DEFAULT_DIMENSION_ORDER,
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
    tile_identifiers = [
        TileIdentifier(fov_id, selector[Axes.ROUND], selector[Axes.CH], selector[Axes.ZPLANE])
        for fov_id in fovs
        for selector in ordered_iterator(axes_sizes)
    ]

    return build_irregular_image(tile_identifiers, image_fetcher, default_shape)


def write_irregular_experiment_json(
        path: str,
        tile_format: ImageFormat,
        *,
        image_tile_identifiers: Mapping[str, Iterable[TileIdentifier]],
        tile_fetchers: Mapping[str, TileFetcher],
        postprocess_func: Optional[Callable[[dict], dict]]=None,
        default_shape: Optional[Mapping[Axes, int]]=None,
        fov_path_generator: Callable[[Path, str], Path] = None,
        tile_opener: Optional[Callable[[Path, Tile, str], BinaryIO]] = None,
        writer_contract: Optional[WriterContract] = None,
) -> None:
    """
    Build and returns a top-level experiment description with the following characteristics:

    Parameters
    ----------
    path : str
        Directory to write the files to.
    tile_format : ImageFormat
        File format to write the tiles as.
    image_tile_identifiers : Mapping[str, Iterable[TileIdentifier]]
        Dictionary mapping the image type to an iterable of TileIdentifiers.
    tile_fetchers : Mapping[str, TileFetcher]
        Dictionary mapping the image type to a TileFetcher.
    postprocess_func : Optional[Callable[[dict], dict]]
        If provided, this is called with the experiment document for any postprocessing.
        An example of this would be to add something to one of the top-level extras field.
        The callable should return what is to be written as the experiment document.
    default_shape : Optional[Tuple[int, int]] (default = None)
        Default shape for the tiles in this experiment.
    fov_path_generator : Optional[Callable[[Path, str], Path]]
        Generates the path for a FOV's json file.  If one is not provided, the default generates
        the FOV's json file at the same level as the top-level json file for an image.    If this is
        not provided, a reasonable default will be provided.  If this is provided, writer_contract
        should not be provided.  This parameter is deprecated and `writer_contract` should be used
        instead.
    tile_opener : Optional[Callable[[Path, Tile, str], BinaryIO]]
        Callable that gets invoked with the following arguments: 1. the directory of the experiment
        that is being constructed, 2. the tile that is being written, and 3. the file extension
        that the tile should be written with.  The callable is expected to return an open file
        handle.  If this is not provided, a reasonable default will be provided.  If this is
        provided, `writer_contract` should not be provided.  This parameter is deprecated and
        `writer_contract` should be used instead.
    writer_contract : Optional[WriterContract]
        Contract for specifying how the slicedimage image is to be laid out.  If this is provided,
        `fov_path_generator` and `tile_opener` should not be provided.
    """
    if postprocess_func is None:
        postprocess_func = lambda doc: doc
    if fov_path_generator is not None or tile_opener is not None:
        warnings.warn(
            "`fov_path_generator` and `tile_opener` options for writing experiment files is "
            "deprecated.  Use `writer_contract` instead.",
            DeprecationWarning)
        if writer_contract is not None:
            raise ValueError(
                "Cannot specify both `writer_contract` and `fov_path_generator` or `tile_opener`")

    experiment_doc: Dict[str, Any] = {
        'version': str(CURRENT_VERSION),
        'images': {},
        'extras': {},
    }
    for image_type, tile_identifiers in image_tile_identifiers.items():
        tile_fetcher = tile_fetchers[image_type]

        image = build_irregular_image(tile_identifiers, tile_fetcher, default_shape)

        Writer.write_to_path(
            image,
            Path(path) / f"{image_type}.json",
            pretty=True,
            partition_path_generator=fov_path_generator,
            tile_opener=tile_opener,
            writer_contract=writer_contract,
            tile_format=tile_format,
        )
        experiment_doc['images'][image_type] = f"{image_type}.json"

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
        fov_path_generator: Optional[Callable[[Path, str], Path]] = None,
        tile_opener: Optional[Callable[[Path, Tile, str], BinaryIO]] = None,
        writer_contract: Optional[WriterContract] = None,
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
    fov_path_generator : Optional[Callable[[Path, str], Path]]
        Generates the path for a FOV's json file.  If one is not provided, the default generates
        the FOV's json file at the same level as the top-level json file for an image.    If this is
        not provided, a reasonable default will be provided.  If this is provided, writer_contract
        should not be provided.
    tile_opener : Optional[Callable[[Path, Tile, str], BinaryIO]]
        Callable that gets invoked with the following arguments: 1. the directory of the experiment
        that is being constructed, 2. the tile that is being written, and 3. the file extension
        that the tile should be written with.  The callable is expected to return an open file
        handle.  If this is not provided, a reasonable default will be provided.  If this is
        provided, writer_contract should not be provided.
    writer_contract : Optional[WriterContract]
        Contract for specifying how the slicedimage image is to be laid out.  If this is provided,
        fov_path_generator and tile_opener should not be provided.
    """
    all_tile_fetcher: MutableMapping[str, TileFetcher] = {}
    if aux_tile_fetcher is not None:
        all_tile_fetcher.update(aux_tile_fetcher)
    if primary_tile_fetcher is not None:
        all_tile_fetcher[FieldOfView.PRIMARY_IMAGES] = primary_tile_fetcher

    image_tile_identifiers: MutableMapping[str, Iterable[TileIdentifier]] = dict()
    image_tile_fetchers: MutableMapping[str, TileFetcher] = dict()

    dimension_cardinality_of_images = {FieldOfView.PRIMARY_IMAGES: primary_image_dimensions}
    for image_type, image_dimension in aux_name_to_dimensions.items():
        dimension_cardinality_of_images[image_type] = image_dimension

    for image_type, dimension_cardinality_of_image in dimension_cardinality_of_images.items():
        axes_sizes = join_axes_labels(
            dimension_order,
            rounds=range(dimension_cardinality_of_image[Axes.ROUND]),
            chs=range(dimension_cardinality_of_image[Axes.CH]),
            zplanes=range(dimension_cardinality_of_image[Axes.ZPLANE]),
        )
        image_tile_identifiers[image_type] = [
            TileIdentifier(fov_id, selector[Axes.ROUND], selector[Axes.CH], selector[Axes.ZPLANE])
            for fov_id in range(fov_count)
            for selector in ordered_iterator(axes_sizes)
        ]

        image_tile_fetchers[image_type] = all_tile_fetcher.get(
            image_type, tile_fetcher_factory(RandomNoiseTile))

    return write_irregular_experiment_json(
        path, tile_format,
        image_tile_identifiers=image_tile_identifiers,
        tile_fetchers=image_tile_fetchers,
        postprocess_func=postprocess_func,
        default_shape=default_shape,
        fov_path_generator=fov_path_generator,
        tile_opener=tile_opener,
        writer_contract=writer_contract,
    )
