import copy
import json
import pprint
from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Optional,
    Sequence,
    Set,
    Union
)

from semantic_version import Version
from slicedimage import Collection, TileSet
from slicedimage.io import Reader, resolve_path_or_url, resolve_url
from slicedimage.urlpath import pathjoin

from starfish.codebook.codebook import Codebook
from starfish.config import StarfishConfig
from starfish.imagestack.imagestack import ImageStack
from starfish.imagestack.parser.crop import CropParameters
from starfish.spacetx_format import validate_sptx
from starfish.types import Axes, Coordinates
from .version import MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION


class FieldOfView:
    """
    This encapsulates a field of view.  It contains the primary image and auxiliary images that are
    associated with the field of view.

    All images can be accessed using a the get_image('primary') method with the name of the image
    type. The primary image is accessed using the name
    :py:attr:`starfish.experiment.experiment.FieldOFView.PRIMARY_IMAGES`.

    Access a FOV through a experiment. experiement.fov()

    Attributes
    ----------
    name : str
        The name of the FOV.
    image_types : Set[str]
        A set of all the image types.
    """

    PRIMARY_IMAGES = 'primary'

    def __init__(
            self, name: str,
            image_tilesets: MutableMapping[str, TileSet]
    ) -> None:
        """
        Fields of views can obtain their primary image from either an ImageStack or a TileSet (but
        only one).  It can obtain their auxiliary image dictionary from either a dictionary of
        auxiliary image name to ImageStack or a dictionary of auxiliary image name to TileSet (but
        only one).

        Note that if the source image is from a TileSet, the decoding of TileSet to ImageStack does
        not happen until the image is accessed.  Be prepared to handle errors when images are
        accessed.
        """
        self._images: MutableMapping[str, TileSet] = dict()
        self._name = name
        self.aligned_coordinate_groups: Dict[str, List[CropParameters]] = dict()
        for name, tileset in image_tilesets.items():
            self.aligned_coordinate_groups[name] = self.parse_coordinate_groups(tileset)
        self._images = image_tilesets

    def __repr__(self):
        images = '\n    '.join(
            f'{k}: {v}'
            for k, v in self._images.items()
            if k != FieldOfView.PRIMARY_IMAGES
        )
        return (
            f"<starfish.FieldOfView>\n"
            f"  Primary Image: {self._images[FieldOfView.PRIMARY_IMAGES]}\n"
            f"  Auxiliary Images:\n"
            f"    {images}"
        )

    def parse_coordinate_groups(self, tileset: TileSet) -> List[CropParameters]:
        """Takes a tileset and compares the physical coordinates on each tile to
         create aligned coordinate groups (groups of tiles that have the same physical coordinates)

         Returns
         -------
         A list of CropParameters. Each entry describes the r/ch/z values of tiles that are aligned
         (have matching coordinates)
         """
        coord_groups: OrderedDict[tuple, CropParameters] = OrderedDict()
        for tile in tileset.tiles():
            x_y_coords = (
                tile.coordinates[Coordinates.X][0], tile.coordinates[Coordinates.X][1],
                tile.coordinates[Coordinates.Y][0], tile.coordinates[Coordinates.Y][1]
            )
            # A tile with this (x, y) has already been seen, add tile's Indices to CropParameters
            if x_y_coords in coord_groups:
                crop_params = coord_groups[x_y_coords]
                crop_params._add_permitted_axes(Axes.CH, tile.indices[Axes.CH])
                crop_params._add_permitted_axes(Axes.ROUND, tile.indices[Axes.ROUND])
                if Axes.ZPLANE in tile.indices:
                    crop_params._add_permitted_axes(Axes.ZPLANE, tile.indices[Axes.ZPLANE])
            else:
                coord_groups[x_y_coords] = CropParameters(
                    permitted_chs=[tile.indices[Axes.CH]],
                    permitted_rounds=[tile.indices[Axes.ROUND]],
                    permitted_zplanes=[tile.indices[Axes.ZPLANE]] if Axes.ZPLANE in tile.indices
                    else None)
        return list(coord_groups.values())

    @property
    def name(self) -> str:
        return self._name

    @property
    def image_types(self) -> Set[str]:
        return set(self._images.keys())

    def show_aligned_image_groups(self) -> None:
        """
        Describe the aligned subgroups for each Tileset in this FOV

        ex.
        {'nuclei': ' Group 0:  <starfish.ImageStack r={0}, ch={0}, z={0}, (y, x)=(190,270)>',
        'primary': ' Group 0:  <starfish.ImageStack r={0, 1, 2, 3, 4, 5}, ch={0, 1, '
        '2}, z={0}, (y, x)=(190, 270)>'}

        Means there are two tilesets in this FOV (primary and nuclei), and because all images have
        the same (x, y) coordinates, each tileset has a single aligned subgroup.
        """
        all_groups = dict()
        for name, groups in self.aligned_coordinate_groups.items():
            y_size = self._images[name].default_tile_shape[0]
            x_size = self._images[name].default_tile_shape[1]
            info = '\n'.join(
                f" Group {k}: "
                f" <starfish.ImageStack "
                f"r={v._permitted_rounds if v._permitted_rounds else 1}, "
                f"ch={v._permitted_chs if v._permitted_chs else 1}, "
                f"z={v._permitted_zplanes if v._permitted_zplanes else 1}, "
                f"(y, x)={y_size, x_size}>"
                for k, v in enumerate(groups)
            )
            all_groups[name] = f'{info}'
        pprint.pprint(all_groups)

    def iterate_image_type(self, image_type: str) -> Iterator[ImageStack]:
        for aligned_group, _ in enumerate(self.aligned_coordinate_groups[image_type]):
            yield self.get_image(item=image_type, aligned_group=aligned_group)

    def get_image(self, item: str, aligned_group: int = 0,
                  x_slice: Optional[Union[int, slice]] = None,
                  y_slice: Optional[Union[int, slice]] = None,
                  ) -> ImageStack:
        """
        Parameters
        ----------

        item: str
            The name of the tileset ex. 'primary' or 'nuclei'
        aligned_group: int
            The aligned subgroup, default 0
        x_slice: int or slice
            The cropping parameters for the x axis
        y_slice:
            The cropping parameters for the y axis

        Returns
        -------
        The instantiated ImageStack
        """
        crop_params = copy.copy((self.aligned_coordinate_groups[item][aligned_group]))
        crop_params._x_slice = x_slice
        crop_params._y_slice = y_slice
        return ImageStack.from_tileset(self._images[item], crop_parameters=crop_params)


class Experiment:
    """
    This encapsulates an experiment, with one or more fields of view and a codebook.  An individual
    FOV can be retrieved using a key, i.e., experiment[fov_name].

    Methods
    -------
    from_json()
        Given a URL or a path to an experiment.json document, return an Experiment object
        corresponding to the document.
    fov()
        Given a callable that accepts a FOV, return the first FOVs that the callable returns True
        when passed the FOV.  Because there is no guaranteed sorting for the FOVs, use this
        cautiously.
    fovs()
        Given a callable that accepts a FOV, return all the FOVs that the callable returns True when
        passed the FOV.
    fovs_by_name()
        Given one or more FOV names, return the FOVs that match those names.

    Attributes
    ----------
    codebook : Codebook
        Returns the codebook associated with this experiment.
    extras : Dict
        Returns the extras dictionary associated with this experiment.
    """
    def __init__(
            self,
            fovs: Sequence[FieldOfView],
            codebook: Codebook,
            extras: dict,
            *,
            src_doc: dict=None,
    ) -> None:
        self._fovs = fovs
        self._codebook = codebook
        self._extras = extras
        self._src_doc = src_doc

    def __repr__(self):

        # truncate the list of fields of view if it is longer than print_n_fov
        print_n_fov = 4
        n_fields_of_view = list(self.items())[:print_n_fov]
        fields_of_view_str = "\n".join(
            f'{k}: {v}' for k, v in n_fields_of_view
        )

        # add an ellipsis if not all fields of view are being printed
        if len(self._fovs) > print_n_fov:
            fov_repr = f"{{\n{fields_of_view_str}\n  ...,\n}}"
        else:
            fov_repr = f"{{\n{fields_of_view_str}\n}}"

        # return the formatted string
        object_repr = f"<starfish.Experiment (FOVs={len(self._fovs)})>\n"
        return object_repr + fov_repr

    @classmethod
    def from_json(cls, json_url: str) -> "Experiment":
        """
        Construct an `Experiment` from an experiment.json file format specifier.
        Loads configuration from StarfishConfig.

        Parameters
        ----------
        json_url : str
            file path or web link to an experiment.json file

        Returns
        -------
        Experiment :
            Experiment object serving the requested experiment data

        """

        config = StarfishConfig()

        if config.strict:
            valid = validate_sptx.validate(json_url)
            if not valid:
                raise Exception("validation failed")

        backend, name, baseurl = resolve_path_or_url(json_url, config.slicedimage)
        with backend.read_contextmanager(name) as fh:
            experiment_document = json.load(fh)

        version = cls.verify_version(experiment_document['version'])

        _, codebook_name, codebook_baseurl = resolve_url(experiment_document['codebook'],
                                                         baseurl, config.slicedimage)
        codebook_absolute_url = pathjoin(codebook_baseurl, codebook_name)
        codebook = Codebook.from_json(codebook_absolute_url)

        extras = experiment_document['extras']

        fovs: MutableSequence[FieldOfView] = list()
        fov_tilesets: MutableMapping[str, TileSet]
        if version < Version("5.0.0"):
            primary_image: Collection = Reader.parse_doc(experiment_document['primary_images'],
                                                         baseurl, config.slicedimage)
            auxiliary_images: MutableMapping[str, Collection] = dict()
            for aux_image_type, aux_image_url in experiment_document['auxiliary_images'].items():
                auxiliary_images[aux_image_type] = Reader.parse_doc(
                    aux_image_url, baseurl, config.slicedimage)

            for fov_name, primary_tileset in primary_image.all_tilesets():
                fov_tilesets = dict()
                fov_tilesets[FieldOfView.PRIMARY_IMAGES] = primary_tileset
                for aux_image_type, aux_image_collection in auxiliary_images.items():
                    aux_image_tileset = aux_image_collection.find_tileset(fov_name)
                    if aux_image_tileset is not None:
                        fov_tilesets[aux_image_type] = aux_image_tileset

                fov = FieldOfView(fov_name, image_tilesets=fov_tilesets)
                fovs.append(fov)
        else:
            images: MutableMapping[str, Collection] = dict()
            all_fov_names: MutableSet[str] = set()
            for image_type, image_url in experiment_document['images'].items():
                image = Reader.parse_doc(image_url, baseurl, config.slicedimage)
                images[image_type] = image
                for fov_name, _ in image.all_tilesets():
                    all_fov_names.add(fov_name)

            for fov_name in all_fov_names:
                fov_tilesets = dict()
                for image_type, image_collection in images.items():
                    image_tileset = image_collection.find_tileset(fov_name)
                    if image_tileset is not None:
                        fov_tilesets[image_type] = image_tileset

                fov = FieldOfView(fov_name, image_tilesets=fov_tilesets)
                fovs.append(fov)

        return Experiment(fovs, codebook, extras, src_doc=experiment_document)

    @classmethod
    def verify_version(cls, semantic_version_str: str) -> Version:
        version = Version(semantic_version_str)
        if not (MIN_SUPPORTED_VERSION <= version <= MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"version {version} not supported.  This version of the starfish library only "
                f"supports formats from {MIN_SUPPORTED_VERSION} to "
                f"{MAX_SUPPORTED_VERSION}")
        return version

    def fov(
            self,
            filter_fn: Callable[[FieldOfView], bool]=lambda _: True,
            key_fn: Callable[[FieldOfView], str]=lambda fov: fov.name,
    ) -> FieldOfView:
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return the first
        FOV such that filter_fn(FOV) returns True. The order of the filtered FOVs will be determined
        by the key_fn callable. By default, this matches the order of fov.name.

        If no FOV matches, raise LookupError.
        """
        for fov in sorted(self._fovs, key=key_fn):
            if filter_fn(fov):
                return fov
        raise LookupError("Cannot find any FOV that the filter allows.")

    def fovs(
            self,
            filter_fn: Callable[[FieldOfView], bool]=lambda _: True,
            key_fn: Callable[[FieldOfView], str]=lambda fov: fov.name,
    ) -> Sequence[FieldOfView]:
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return a list of
        FOVs such that filter_fn(FOV) returns True. The returned list is sorted based on the key_fn
        callable, which by default matches the order of fov.name.
        """
        results: MutableSequence[FieldOfView] = list()
        for fov in self._fovs:
            if not filter_fn(fov):
                continue

            results.append(fov)
        results = sorted(results, key=key_fn)
        return results

    def fovs_by_name(
        self,
        *names,
        key_fn: Callable[[FieldOfView], str]=lambda fov: fov.name,
    ) -> Sequence[FieldOfView]:
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return a list of
        FOVs such that filter_fn(FOV) returns True.  The returned list is sorted based on the key_fn
        callable, which by default matches the order of fov.name.
        """
        return self.fovs(filter_fn=lambda fov: fov.name in names)

    def __getitem__(self, item):
        fovs = self.fovs_by_name(item)
        if len(fovs) == 0:
            raise IndexError(f"No field of view with name \"{item}\"")
        return fovs[0]

    def keys(self):
        return (fov.name for fov in self.fovs())

    def values(self):
        return (fov for fov in self.fovs())

    def items(self):
        return ((fov.name, fov) for fov in self.fovs())

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    @property
    def extras(self):
        return self._extras
