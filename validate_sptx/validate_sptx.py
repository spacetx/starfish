import json
import os
import posixpath
import sys
from pkg_resources import resource_filename

import click
from slicedimage.io import resolve_path_or_url

from .util import SpaceTxValidator


def _get_absolute_schema_path(schema_name: str) -> str:
    """turn the name of the schema into an absolute path by joining it to <package_root>/schema."""
    return resource_filename("validate_sptx", posixpath.join("schema", schema_name))


@click.command()
@click.option('--experiment-json', help='image metadata file to validate')
def validate_sptx(experiment_json: str) -> None:
    """validate a spaceTx formatted experiment.
    Accepts local filepaths or files hosted at http links.
    """

    valid = True

    # use slicedimage to read the top-level experiment json file passed by the user
    backend, name, baseurl = resolve_path_or_url(experiment_json)
    with backend.read_contextmanager(name) as fh:
        experiment = json.load(fh)

    # validate experiment.json
    experiment_validator = SpaceTxValidator(_get_absolute_schema_path('experiment.json'))
    valid &= experiment_validator.validate_object(experiment)

    # validate manifests that it links to.
    possible_manifests = []
    manifest_validator = SpaceTxValidator(_get_absolute_schema_path('fov_manifest.json'))
    with backend.read_contextmanager(experiment['primary_images']) as fh:
        possible_manifests.append(json.load(fh))

    # loop over all the manifests that are stored in auxiliary images. Disallowed names will
    # have already been excluded by experiment validation.
    for manifest in experiment['auxiliary_images'].values():
        with backend.read_contextmanager(manifest) as fh:
            possible_manifests.append(json.load(fh))

    # we allow the objects linked from primary_images and auxiliary images to either be
    # manifests OR field_of_view files. We distinguish these by checking if they have a `contents`
    # flag, which indicates it is a manifest.
    fovs = []
    for manifest in possible_manifests:
        if 'contents' in manifest:  # is a manifest; validate
            valid &= manifest_validator.validate_object(manifest)

            # contains fields of view
            for key, fov in manifest['contents'].items():
                with backend.read_contextmanager(fov) as fh:
                    fovs.append(json.load(fh))

        else:  # manifest is a field of view
            fovs.append(manifest)

    # validate fovs
    assert len(fovs) != 0
    fov_validator = SpaceTxValidator(_get_absolute_schema_path('field_of_view/field_of_view.json'))
    for fov in fovs:
        valid &= fov_validator.validate_object(fov)

    # validate codebook
    codebook_validator = SpaceTxValidator(_get_absolute_schema_path('codebook/codebook.json'))
    with backend.read_contextmanager(experiment['codebook']) as fh:
        codebook = json.load(fh)
    valid &= codebook_validator.validate_object(codebook)

    if valid:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    validate_sptx()
