import json
import os
from typing import Tuple

from slicedimage import (
    Collection,
    ImageFormat,
    Tile,
    TileSet,
    Writer,
)

from starfish.experiment import Experiment
from starfish.types import Coordinates, Indices
from .imagedata import FetchedTile, RandomNoiseTile, RandomNoiseTileFetcher, TileFetcher


AUX_IMAGE_NAMES = {
    'nuclei',
    'dots',
}


def tile_opener(toc_path, tile, file_ext):
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


def tile_writer(tile, fh):
    tile.copy(fh)
    return ImageFormat.TIFF


def fov_path_generator(parent_toc_path, toc_name):
    toc_basename = os.path.splitext(os.path.basename(parent_toc_path))[0]
    return os.path.join(
        os.path.dirname(parent_toc_path),
        "{}-{}.json".format(toc_basename, toc_name),
    )


def build_image(
        fov_count, hyb_count, ch_count, z_count,
        image_fetcher: TileFetcher,
        default_shape: Tuple[int, int]
):
    """
    Build and returns an image set with the following characteristics:

    Parameters
    ----------
    fov_count : int
        Number of fields of view in this image set.
    hyb_count : int
        Number for hybridizations in this image set.
    ch_count : int
        Number for channels in this image set.
    z_count : int
        Number of z-layers in this image set.
    default_shape : Tuple[int, int]
        Default shape of the individual tiles in this image set.

    Returns
    -------
    The slicedimage collection representing the image.
    """
    collection = Collection()
    for fov_ix in range(fov_count):
        fov_images = TileSet(
            [Indices.X, Indices.Y, Indices.Z, Indices.ROUND, Indices.CH],
            {Indices.ROUND: hyb_count, Indices.CH: ch_count, Indices.Z: z_count},
            default_shape,
            ImageFormat.TIFF,
        )

        for z_ix in range(z_count):
            for hyb_ix in range(hyb_count):
                for ch_ix in range(ch_count):
                    image = image_fetcher.get_tile(fov_ix, hyb_ix, ch_ix, z_ix)
                    tile = Tile(
                        {
                            Coordinates.X: (0.0, 0.0001),
                            Coordinates.Y: (0.0, 0.0001),
                            Coordinates.Z: (0.0, 0.0001),
                        },
                        {
                            Indices.Z: z_ix,
                            Indices.ROUND: hyb_ix,
                            Indices.CH: ch_ix,
                        },
                        image.shape,
                    )
                    tile.set_source_fh_contextmanager(image.tile_data_handle, image.format)
                    fov_images.add_tile(tile)
        collection.add_partition("fov_{:03}".format(fov_ix), fov_images)
    return collection


def write_experiment_json(
        path,
        fov_count,
        hyb_dimensions,
        aux_name_to_dimensions,
        primary_tile_fetcher=None,
        aux_tile_fetcher=None,
        postprocess_func=None,
        default_shape=(1536, 1024),
):
    """
    Build and returns a top-level experiment description with the following characteristics:

    Parameters
    ----------
    path : str
        Directory to write the files to.
    fov_count : int
        Number of fields of view in this experiment.
    hyb_dimensions : Mapping[str, int]
        Dictionary mapping dimension name to dimension size for the hybridization image.
    aux_name_to_dimensions : Mapping[str, Mapping[str, int]]
        Dictionary mapping the auxiliary image type to dictionaries, which map from dimension name
        to dimension size.
    primary_tile_fetcher : Optional[ImageFetcher]
        ImageFetcher for hybridization images.  Set this if you want specific image data to be set
        for the hybridization images.  If not provided, the image data is set to random noise via
        :class:`RandomNoiseImageFetcher`.
    aux_tile_fetcher : Optional[Mapping[str, ImageFetcher]]
        ImageFetchers for auxiliary images.  Set this if you want specific image data to be set for
        one or more aux image types.  If not provided for any given aux image, the image data is
        set to random noise via :class:`RandomNoiseImageFetcher`.
    postprocess_func : Optional[Callable[[dict], dict]]
        If provided, this is called with the experiment document for any postprocessing.
        An example of this would be to add something to one of the top-level extras field.
        The callable should return what is to be written as the experiment document.
    default_shape : Tuple[int, int] (default = (1536, 1024))
        Default shape for the tiles in this experiment.
    """
    if primary_tile_fetcher is None:
        primary_tile_fetcher = RandomNoiseTileFetcher()
    if aux_tile_fetcher is None:
        aux_tile_fetcher = {}
    if postprocess_func is None:
        postprocess_func = lambda doc: doc

    experiment_doc = {
        'version': str(Experiment.CURRENT_VERSION),
        'auxiliary_images': {},
        'extras': {},
    }
    hybridization_image = build_image(
        fov_count,
        hyb_dimensions[Indices.ROUND], hyb_dimensions[Indices.CH], hyb_dimensions[Indices.Z],
        primary_tile_fetcher,
        default_shape=default_shape,
    )
    Writer.write_to_path(
        hybridization_image,
        os.path.join(path, "hybridization.json"),
        pretty=True,
        partition_path_generator=fov_path_generator,
        tile_opener=tile_opener,
        tile_writer=tile_writer,
    )
    experiment_doc['hybridization_images'] = "hybridization.json"

    for aux_name, aux_dimensions in aux_name_to_dimensions.items():
        if aux_dimensions is None:
            continue
        auxiliary_image = build_image(
            fov_count,
            aux_dimensions[Indices.ROUND], aux_dimensions[Indices.CH], aux_dimensions[Indices.Z],
            aux_tile_fetcher.get(aux_name, RandomNoiseTileFetcher()),
            default_shape=default_shape,
        )
        Writer.write_to_path(
            auxiliary_image,
            os.path.join(path, "{}.json".format(aux_name)),
            pretty=True,
            partition_path_generator=fov_path_generator,
            tile_opener=tile_opener,
            tile_writer=tile_writer,
        )
        experiment_doc['auxiliary_images'][aux_name] = "{}.json".format(aux_name)

    experiment_doc = postprocess_func(experiment_doc)
    with open(os.path.join(path, "experiment.json"), "w") as fh:
        json.dump(experiment_doc, fh, indent=4)
