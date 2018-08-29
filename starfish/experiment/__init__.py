import json
from typing import Callable, MutableMapping, MutableSequence, Optional, Sequence, Set, Union

from semantic_version import Version
from slicedimage import Collection, TileSet
from slicedimage.io import Reader, resolve_path_or_url, resolve_url
from slicedimage.urlpath import pathjoin

from starfish.codebook import Codebook
from starfish.stack import ImageStack


class FieldOfView:
    """
    This encapsulates a field of view.  It contains the primary image and auxiliary images that are
    associated with the field of view.

    Auxiliary images can be accessed using a key, i.e., FOV[aux_image_type].

    Properties
    -------
    name                   The name of the FOV.  In an experiment with only a single FOV, this may
                           be None.
    primary_image          The primary image for this field of view.
    auxiliary_image_types  A set of all the auxiliary image types.
    """
    def __init__(
            self,
            name: Optional[str],
            primary_image: Optional[ImageStack]=None,
            auxiliary_images: Optional[MutableMapping[str, ImageStack]]=None,
            primary_image_tileset: Optional[TileSet]=None,
            auxiliary_image_tilesets: Optional[MutableMapping[str, TileSet]]=None,
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
        self._primary_image: Union[ImageStack, TileSet]
        self._auxiliary_images: MutableMapping[str, Union[ImageStack, TileSet]]
        if primary_image is not None:
            self._primary_image = primary_image
            if primary_image_tileset is not None:
                raise ValueError(
                    "Only one of (primary_image, primary_image_tileset) should be set.")
        elif primary_image_tileset is not None:
            self._primary_image = primary_image_tileset
        else:
            raise ValueError("Field of view must have a primary image")
        if auxiliary_images is not None:
            self._auxiliary_images = auxiliary_images
            if auxiliary_image_tilesets is not None:
                raise ValueError(
                    "Only one of (auxiliary_images, auxiliary_image_tilesets) should be set.")
        elif auxiliary_image_tilesets is not None:
            self._auxiliary_images = auxiliary_image_tilesets
        else:
            self._auxiliary_images = dict()

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def primary_image(self) -> ImageStack:
        if isinstance(self._primary_image, TileSet):
            self._primary_image = ImageStack(self._primary_image)
        return self._primary_image

    @property
    def auxiliary_image_types(self) -> Set[str]:
        return set(self._auxiliary_images.keys())

    def __getitem__(self, item) -> ImageStack:
        if isinstance(self._auxiliary_images[item], TileSet):
            self._auxiliary_images[item] = ImageStack(self._auxiliary_images[item])
        return self._auxiliary_images[item]


class Experiment:
    """
    This encapsulates an experiment, with one or more fields of view and a codebook.  An individual
    FOV can be retrieved using a key, i.e., experiment[fov_name].

    Constructors
    -------
    from_json     Given a URL or a path to an experiment.json document, return an Experiment object
                  corresponding to the document.

    Methods
    -------
    fov           Given a callable that accepts a FOV, return the first FOVs that the callable
                  returns True when passed the FOV.  Because there is no guaranteed sorting for the
                  FOVs, use this cautiously.
    fovs          Given a callable that accepts a FOV, return all the FOVs that the callable returns
                  True when passed the FOV.
    fovs_by_name  Given one or more FOV names, return the FOVs that match those names.

    Properties
    -------
    codebook      Returns the codebook associated with this experiment.
    extras        Returns the extras dictionary associated with this experiment.
    """
    CURRENT_VERSION = Version("1.0.0")
    MIN_SUPPORTED_VERSION = Version("1.0.0")
    MAX_SUPPORTED_VERSION = Version("1.0.0")

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

    @classmethod
    def from_json(cls, json_url: str) -> "Experiment":
        """
        Construct an `Experiment` from an experiment.json file format specifier

        Parameters
        ----------
        json_url : str
            file path or web link to an experiment.json file

        Returns
        -------
        Experiment :
            Experiment object serving the requested experiment data

        """
        backend, name, baseurl = resolve_path_or_url(json_url)
        with backend.read_contextmanager(name) as fh:
            experiment_document = json.load(fh)

        cls.verify_version(experiment_document['version'])

        _, codebook_name, codebook_baseurl = resolve_url(experiment_document['codebook'], baseurl)
        codebook_absolute_url = pathjoin(codebook_baseurl, codebook_name)
        codebook = Codebook.from_json(codebook_absolute_url)

        extras = experiment_document['extras']

        primary_image: Union[Collection, TileSet] = Reader.parse_doc(
            experiment_document['primary_images'], baseurl)
        auxiliary_images: MutableMapping[str, Union[Collection, TileSet]] = dict()
        for aux_image_type, aux_image_url in experiment_document['auxiliary_images'].items():
            auxiliary_images[aux_image_type] = Reader.parse_doc(aux_image_url, baseurl)

        if isinstance(primary_image, TileSet):
            # make sure that all the aux images are also TileSets
            for aux_image_tileset in auxiliary_images.values():
                assert isinstance(aux_image_tileset, TileSet)

            fov = FieldOfView(
                None,
                primary_image_tileset=primary_image,
                auxiliary_image_tilesets=auxiliary_images
            )
            return Experiment([fov], codebook, extras, src_doc=experiment_document)

        # everything should be Collections
        fovs: MutableSequence[FieldOfView] = list()
        for fov_name, primary_tileset in primary_image.all_tilesets():
            aux_image_tilesets_for_fov: MutableMapping[str, TileSet] = dict()
            for aux_image_type, aux_image_collection in auxiliary_images.items():
                assert isinstance(aux_image_collection, Collection)
                aux_image_tileset = aux_image_collection.find_tileset(fov_name)
                if aux_image_tileset is not None:
                    aux_image_tilesets_for_fov[aux_image_type] = aux_image_tileset

            fov = FieldOfView(
                fov_name,
                primary_image_tileset=primary_tileset,
                auxiliary_image_tilesets=aux_image_tilesets_for_fov,
            )
            fovs.append(fov)

        return Experiment(fovs, codebook, extras, src_doc=experiment_document)

    @classmethod
    def verify_version(cls, semantic_version_str: str) -> None:
        version = Version(semantic_version_str)
        if not (Experiment.MIN_SUPPORTED_VERSION <= version <= Experiment.MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"version {version} not supported.  This version of the starfish library only "
                f"supports formats from {Experiment.MIN_SUPPORTED_VERSION} to "
                f"{Experiment.MAX_SUPPORTED_VERSION}")

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

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    @property
    def extras(self):
        return self._extras
