import json
from typing import (
    Any,
    Callable,
    Collection,
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
from slicedimage import Collection as collec
from slicedimage import TileSet
from slicedimage.io import Reader, resolve_path_or_url, resolve_url
from slicedimage.urlpath import pathjoin

from starfish.core.codebook.codebook import Codebook
from starfish.core.config import StarfishConfig
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.parser.crop import CropParameters
from starfish.core.spacetx_format import validate_sptx
from .version import MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION

_SINGLETON = object()


class FieldOfView:
    """
    This encapsulates a field of view.  It contains the primary image and auxiliary images that are
    associated with the field of view.

    All images can be accessed using
    :py:func:`~starfish.experiment.experiment.FieldOfView.get_image`
    with the name of the image type and any which rounds/chs/zplanes you want to include
    (if none are provided all are included) If the resulting tileset has unaligned tiles
    (their x/y coordinates do not all match) we return a list on ImageStacks where each stack
    represents an aligned group of tiles. The primary image is accessed using the name
    :py:attr:`~starfish.experiment.experiment.FieldOFView.PRIMARY_IMAGES`.

    Notes
    -----
    Field of views obtain their primary image from a :py:class:`~slicedimage.TileSet`.
    They can obtain their auxiliary image dictionary from a dictionary of auxiliary image to
    :py:class:`~slicedimage.TileSet`.

    The decoding of :py:class:`~slicedimage.TileSet` to
    :py:class:`~starfish.imagestack.imagestack.ImageStack`
    does not happen until the image is accessed. ImageStacks can only be initialized with
    aligned tilesets, so if there are multiple you may need to iterate through the groups
    using :py:func:`~starfish.experiment.experiment.FieldOfView.get_images`
    and process each one individually.

    Be prepared to handle errors when images are accessed.

    Access a FieldOfView through Experiment.
    :py:func:`~starfish.experiment.experiment.Experiment.fov`.

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
        self._images: MutableMapping[str, TileSet] = dict()
        self._name = name
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

    @property
    def name(self) -> str:
        return self._name

    @property
    def image_types(self) -> Set[str]:
        return set(self._images.keys())

    def get_image(self, item: str,
                  aligned_group: Any = _SINGLETON,
                  rounds: Optional[Collection[int]] = None,
                  chs: Optional[Collection[int]] = None,
                  zplanes: Optional[Collection[int]] = None,
                  x: Optional[Union[int, slice]] = None,
                  y: Optional[Union[int, slice]] = None,
                  ) -> ImageStack:
        """
        Load into memory the first Imagestack representation of an aligned image group. If crop
        parameters provided, first crop the TileSet.

        Parameters
        ----------
        item: str
            The name of the tileset ex. 'primary' or 'nuclei'
        rounds : Optional[Collection[int]]
            The rounds in the original dataset to load into the ImageStack.  If this is not set,
            then all rounds are loaded into the ImageStack.
        chs : Optional[Collection[int]]
            The channels in the original dataset to load into the ImageStack.  If this is not set,
            then all channels are loaded into the ImageStack.
        zplanes : Optional[Collection[int]]
            The z-layers in the original dataset to load into the ImageStack.  If this is not set,
            then all z-layers are loaded into the ImageStack.
        x : Optional[Union[int, slice]]
            The x-range in the x-y tile that is loaded into the ImageStack.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.
        y : Optional[Union[int, slice]]
            The y-range in the x-y tile that is loaded into the ImageStack.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.

        Returns
        -------
        ImageStack
            The instantiated image stack

        """
        if aligned_group != _SINGLETON:
            raise RuntimeError("The parameter 'aligned_group` is no longer accepted. Please use"
                               "FieldOfView.get_images() and  provide sets of selected axes "
                               "instead")
        stack_iterator = self.get_images(item=item, rounds=rounds,
                                         chs=chs, zplanes=zplanes, x=x, y=y)
        return next(stack_iterator)

    def get_images(self, item: str,
                   rounds: Optional[Collection[int]] = None,
                   chs: Optional[Collection[int]] = None,
                   zplanes: Optional[Collection[int]] = None,
                   x: Optional[Union[int, slice]] = None,
                   y: Optional[Union[int, slice]] = None,
                   ) -> Iterator[ImageStack]:
        """
        Load into memory the an iterator of aligned Imagestacks for the given tileset and selected
        axes.

        Parameters
        ----------
        item: str
            The name of the tileset ex. 'primary' or 'nuclei'
        rounds : Optional[Collection[int]]
            The rounds in the original dataset to load into the ImageStack/s.  If this is not set,
            then all rounds are loaded into the ImageStack.
        chs : Optional[Collection[int]]
            The channels in the original dataset to load into the ImageStac/s.  If this is not set,
            then all channels are loaded into the ImageStack.
        zplanes : Optional[Collection[int]]
            The z-layers in the original dataset to load into the ImageStack/s.  If this is not set,
            then all z-layers are loaded into the ImageStack.
        x : Optional[Union[int, slice]]
            The x-range in the x-y tile that is loaded into the ImageStack/s.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.
        y : Optional[Union[int, slice]]
            The y-range in the x-y tile that is loaded into the ImageStack/s.  If this is not set,
            then the entire x-y tile is loaded into the ImageStack.

        Returns
        -------
        AlignedImageStackIterator
            The instantiated ImageStack or list of Imagestacks if the parameters given include
            multiple aligned groups.

        """
        # Parse the tileset into aligned groups, only include tiles from selected axes.
        aligned_groups = CropParameters.parse_aligned_groups(self._images[item],
                                                             rounds=rounds, chs=chs,
                                                             zplanes=zplanes,
                                                             x=x, y=y)
        aligned_stack_iterator = AlignedImageStackIterator(tileset=self._images[item],
                                                           aligned_groups=aligned_groups)
        return aligned_stack_iterator


class AlignedImageStackIterator(Iterator[ImageStack]):
    """Iterator class of AlignedImageStacks."""
    def __init__(self, tileset: TileSet, aligned_groups: List[CropParameters]):
        self.size = len(aligned_groups)
        self.aligned_groups = iter(aligned_groups)
        self.tileset = tileset

    def __len__(self) -> int:
        return self.size

    def __next__(self) -> ImageStack:
        aligned_group = next(self.aligned_groups)
        stack = ImageStack.from_tileset(self.tileset, aligned_group)
        return stack


class Experiment:
    """
    Encapsulates an experiment, with one or more fields of view and a
    :py:class:`~starfish.codebook.codebook.Codebook`. An individual
    :py:class:`~starfish.experiment.experiment.FieldOfView` can be retrieved using a
    key, i.e., experiment[fov_name].

    Attributes
    ----------
    codebook : Codebook
        The codebook associated with this experiment.
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
        Construct an Experiment from an experiment.json file format specifier.
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
        codebook = Codebook.open_json(codebook_absolute_url)

        extras = experiment_document['extras']

        fovs: MutableSequence[FieldOfView] = list()
        fov_tilesets: MutableMapping[str, TileSet]
        if version < Version("5.0.0"):
            primary_image: collec = Reader.parse_doc(experiment_document['primary_images'],
                                                     baseurl, config.slicedimage)
            auxiliary_images: MutableMapping[str, collec] = dict()
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
            images: MutableMapping[str, collec] = dict()
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
        """Verifies the compatibility of the current starfish version with
        the experiment version"""
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
        Given a callable filter_fn, apply it to all the FOVs in this experiment.

        Parameters
        ----------
        filter_fn : Callable
            Filter to apply to the list of FOVs
        key_fn : Callable
            The key that determines the order of filtered FOVs, default fov.name

        Returns
        -------
        FieldOfView
            The first FOV that fulfills the filter parameters.

        Raises
        ------
        LookupError :
             If no FOV matches

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
        Given a callable filter_fn, apply it to all the FOVs in this experiment.

        Parameters
        ----------
        filter_fn : Callable
            Filter to apply to the list of FOVs
        key_fn : Callable
            The key that determines the order of filtered FOVs, default fov.name

        Returns
        -------
        Sequence[FieldOfView]
            All fovs that pass the filter function.

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
    ) -> Sequence[FieldOfView]:
        """
        Given a name or set of names, return all fovs that match.

        Parameters
        ----------
        names : str
            The fov names to search for.

        Returns
        -------
        Sequence[FieldOfView]
            All fovs that match the given names.
        """
        return self.fovs(filter_fn=lambda fov: fov.name in names)

    def __getitem__(self, item):
        fovs = self.fovs_by_name(item)
        if len(fovs) == 0:
            raise IndexError(f"No field of view with name \"{item}\"")
        return fovs[0]

    def keys(self):
        """Return all fov names in the experiment"""
        return (fov.name for fov in self.fovs())

    def values(self):
        """Return all FieldOfViews in the experiment"""
        return (fov for fov in self.fovs())

    def items(self):
        """Return all names and fovs in the experiment"""
        return ((fov.name, fov) for fov in self.fovs())

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    @property
    def extras(self):
        return self._extras
