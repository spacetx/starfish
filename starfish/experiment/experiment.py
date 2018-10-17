import json
import os
from typing import (
    Callable,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Optional,
    Sequence,
    Set,
    Union,
)

from semantic_version import Version
from slicedimage import Collection, TileSet
from slicedimage.io import Reader, resolve_path_or_url, resolve_url
from slicedimage.urlpath import pathjoin

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from validate_sptx import validate_sptx
from .version import MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION


class FieldOfView:
    """
    This encapsulates a field of view.  It contains the primary image and auxiliary images that are
    associated with the field of view.

    All images can be accessed using a key, i.e., FOV[aux_image_type].  The primary image is
    accessed using the key :py:attr:`starfish.experiment.experiment.FieldOFView.PRIMARY_IMAGES`.

    Attributes
    ----------
    name : str
        The name of the FOV.
    image_types : Set[str]
        A set of all the image types.
    """

    PRIMARY_IMAGES = 'primary'

    def __init__(
            self,
            name: str,
            images: Optional[MutableMapping[str, ImageStack]]=None,
            image_tilesets: Optional[MutableMapping[str, TileSet]]=None,
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
        self._name = name
        self._images: MutableMapping[str, Union[ImageStack, TileSet]]
        if images is not None:
            self._images = images
            if image_tilesets is not None:
                raise ValueError(
                    "Only one of (images, image_tilesets) should be set.")
        elif image_tilesets is not None:
            self._images = image_tilesets
        else:
            self._images = dict()

    def __repr__(self):
        images = '\n    '.join(
            f'{k}: {v}'
            for k, v in self._images.items()
            if k != FieldOfView.PRIMARY_IMAGES
        )
        return (
            f"<starfish.FieldOfView>\n"
            f"  Primary Image: {self._image[FieldOfView.PRIMARY_IMAGES]}\n"
            f"  Auxiliary Images:\n"
            f"    {images}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def image_types(self) -> Set[str]:
        return set(self._images.keys())

    def __getitem__(self, item) -> ImageStack:
        if isinstance(self._images[item], TileSet):
            self._images[item] = ImageStack(self._images[item])
        return self._images[item]


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
    def from_json(cls, json_url: str, strict: bool=None) -> "Experiment":
        """
        Construct an `Experiment` from an experiment.json file format specifier

        Parameters
        ----------
        json_url : str
            file path or web link to an experiment.json file
        strict : bool
            if true, then all JSON loaded by this method will be
            passed to the appropriate validator
        STARFISH_STRICT_LOADING :
             This parameter is read from the environment. If set, then all JSON loaded by this
             method will be passed to the appropriate validator. The `strict` parameter to this
             method has priority over the environment variable.

        Returns
        -------
        Experiment :
            Experiment object serving the requested experiment data

        """
        if strict is None:
            strict = "STARFISH_STRICT_LOADING" in os.environ
        if strict:
            valid = validate_sptx.validate(json_url)
            if not valid:
                raise Exception("validation failed")

        backend, name, baseurl = resolve_path_or_url(json_url)
        with backend.read_contextmanager(name) as fh:
            experiment_document = json.load(fh)

        version = cls.verify_version(experiment_document['version'])

        _, codebook_name, codebook_baseurl = resolve_url(experiment_document['codebook'], baseurl)
        codebook_absolute_url = pathjoin(codebook_baseurl, codebook_name)
        codebook = Codebook.from_json(codebook_absolute_url)

        extras = experiment_document['extras']

        fovs: MutableSequence[FieldOfView] = list()
        fov_tilesets: MutableMapping[str, TileSet] = dict()
        if version < Version("5.0.0"):
            primary_image: Collection = Reader.parse_doc(experiment_document['primary_images'],
                                                         baseurl)
            auxiliary_images: MutableMapping[str, Collection] = dict()
            for aux_image_type, aux_image_url in experiment_document['auxiliary_images'].items():
                auxiliary_images[aux_image_type] = Reader.parse_doc(aux_image_url, baseurl)

            for fov_name, primary_tileset in primary_image.all_tilesets():
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
                image = Reader.parse_doc(image_url, baseurl)
                images[image_type] = image
                for fov_name, _ in image.all_tilesets():
                    all_fov_names.add(fov_name)

            for fov_name in all_fov_names:
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
    ) -> FieldOfView:
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return the first
        FOV such that filter_fn(FOV) returns True.  Because there is no guaranteed order for the
        FOVs, use this cautiously.

        If no FOV matches, raise LookupError.
        """
        for fov in self._fovs:
            if filter_fn(fov):
                return fov
        raise LookupError("Cannot find any FOV that the filter allows.")

    def fovs(
            self,
            filter_fn: Callable[[FieldOfView], bool]=lambda _: True,
    ) -> Sequence[FieldOfView]:
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return a list of
        FOVs such that filter_fn(FOV) returns True.
        """
        results: MutableSequence[FieldOfView] = list()
        for fov in self._fovs:
            if not filter_fn(fov):
                continue

            results.append(fov)
        return results

    def fovs_by_name(self, *names):
        """
        Given a callable filter_fn, apply it to all the FOVs in this experiment.  Return a list of
        FOVs such that filter_fn(FOV) returns True.
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
