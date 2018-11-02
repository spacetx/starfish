import json
import posixpath
import sys
from typing import Dict

import click
from pkg_resources import resource_filename
from slicedimage.io import resolve_path_or_url

from .util import SpaceTxValidator


def _get_absolute_schema_path(schema_name: str) -> str:
    """turn the name of the schema into an absolute path by joining it to <package_root>/schema."""
    return resource_filename("sptx_format", posixpath.join("schema", schema_name))


@click.command()
@click.option('--experiment-json', help='image metadata file to validate')
def validate_sptx(experiment_json: str, fuzz: bool=False) -> bool:
    return validate(experiment_json, fuzz)

def validate(experiment_json: str, fuzz: bool=False) -> bool:
    """validate a spaceTx formatted experiment.
    Accepts local filepaths or files hosted at http links.

    Parameters
    ----------
    experiment_json : str
        path or URL to a target json object to be validated against the schema passed to this
        object's constructor
    fuzz : bool
        whether or not to perform element-by-element fuzzing.
        If true, will return true and will *not* use warnings.

    Returns
    -------
    bool :
        True, if object valid or fuzz=True, else False
    """

    valid = True

    # use slicedimage to read the top-level experiment json file passed by the user
    backend, name, baseurl = resolve_path_or_url(experiment_json)
    with backend.read_contextmanager(name) as fh:
        experiment = json.load(fh)

    # validate experiment.json
    experiment_validator = SpaceTxValidator(_get_absolute_schema_path('experiment.json'))
    if fuzz:
        experiment_validator.fuzz_object(experiment, name)
    else:
        valid &= experiment_validator.validate_object(experiment, name)

    # loop over all the manifests that are stored in images. Disallowed names will have already been
    # excluded by experiment validation.
    manifests = []
    for manifest in experiment['images'].values():
        with backend.read_contextmanager(manifest) as fh:
            manifests.append((json.load(fh), manifest))

    fovs = []
    manifest_validator = SpaceTxValidator(_get_absolute_schema_path('fov_manifest.json'))
    for manifest, filename in manifests:
        if fuzz:
            manifest_validator.fuzz_object(manifest, filename)
        else:
            if not manifest_validator.validate_object(manifest, filename):
                valid = False
            else:
                for key, fov in manifest['contents'].items():
                    with backend.read_contextmanager(fov) as fh:
                        fovs.append((json.load(fh), fov))

    # fovs may be empty if the manifests were not valid
    if fovs:
        fov_schema = _get_absolute_schema_path('field_of_view/field_of_view.json')
        fov_validator = SpaceTxValidator(fov_schema)
        for fov, filename in fovs:
            if fuzz:
                fov_validator.fuzz_object(fov, filename)
            else:
                valid &= fov_validator.validate_object(fov, filename)

    # validate codebook
    codebook_validator = SpaceTxValidator(_get_absolute_schema_path('codebook/codebook.json'))
    codebook_file = experiment.get('codebook')
    codebook: Dict = {}
    if codebook_file is not None:
        with backend.read_contextmanager(codebook_file) as fh:
            codebook = json.load(fh)
    if fuzz:
        codebook_validator.fuzz_object(codebook, codebook_file)
    else:
        valid &= codebook_validator.validate_object(codebook, codebook_file)

    return valid


if __name__ == "__main__":
    valid = validate_sptx()

    if valid:
        sys.exit(0)
    else:
        sys.exit(1)
