import json
import sys
from typing import Dict

from slicedimage.backends._base import Backend
from slicedimage.io import resolve_path_or_url

from starfish.core.config import StarfishConfig
from starfish.core.util import click
from .util import (get_schema_path, SpaceTxValidator)


@click.command()
@click.option('--experiment-json', help='image metadata file to validate')
def validate_sptx(experiment_json: str, fuzz: bool=False) -> bool:
    return validate(experiment_json, fuzz)

def validate(experiment_json: str, fuzz: bool=False) -> bool:
    """validate a spaceTx formatted experiment.
    Accepts local filepaths or files hosted at http links.
    Loads configuration from StarfishConfig.

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

    Examples
    --------
    The following will read the experiment json file provided, downloading it if necessary,
    and begin recursively validating it and all referenced json files (e.g. codebook.json):

        >>> from starfish.core.spacetx_format import validate_sptx
        >>> valid = validate_sptx.validate(json_url)
    """

    config = StarfishConfig()
    valid = True

    # use slicedimage to read the top-level experiment json file passed by the user
    try:
        backend, name, baseurl = resolve_path_or_url(
            experiment_json, backend_config=config.slicedimage)
    except ValueError as exception:
        raise Exception(f"could not load {experiment_json}:\n{exception}")

    with backend.read_contextmanager(name) as fh:
        experiment = json.load(fh)

    # validate experiment.json
    valid &= validate_file(name, "experiment.json", fuzz, backend)

    # loop over all the manifests that are stored in images. Disallowed names will have already been
    # excluded by experiment validation.
    for manifest in experiment['images'].values():
        obj: Dict = dict()
        if not validate_file(manifest, "fov_manifest.json", fuzz, backend, obj):
            valid = False
        else:
            for key, fov in obj['contents'].items():
                valid &= validate_file(fov, 'field_of_view/field_of_view.json', fuzz, backend)

    codebook_file = experiment.get('codebook')
    if codebook_file is not None:
        valid &= validate_file(codebook_file, "codebook/codebook.json", fuzz, backend)

    return valid


def validate_file(file: str, schema: str, fuzz: bool=False,
                  backend: Backend=None, output: Dict=None) -> bool:
    """validate a spaceTx formatted file with a given schema.
    Accepts local filepaths or files hosted at http links.

    Parameters
    ----------
    file : str
        path or URL to a target json object to be validated against the schema passed
    schema : str
        resource path to the schema
    backend : slicedimage.backends._base.Backend
        backend previously loaded from a file path or URL,
        or potentially None if a new backend should be loaded.
    fuzz : bool
        whether or not to perform element-by-element fuzzing.
        If true, will return true and will *not* use warnings.
    output : Dict
        dictionary into which the output object can be stored

    Returns
    -------
    bool :
        True, if object valid or fuzz=True, else False

    Examples
    --------
    The following will read the codebook json file provided, downloading it if necessary:

        >>> from starfish.core.spacetx_format.validate_sptx import validate_file
        >>> valid = validate_sptx.validate_file(json_url, "codebook/codebook.json")
    """

    if backend is None:
        backend, name, baseurl = resolve_path_or_url(file)
    else:
        name = file

    with backend.read_contextmanager(name) as fh:
        obj = json.load(fh)
        if output is not None:
            output.update(obj)

    validator = SpaceTxValidator(get_schema_path(schema, obj))

    if fuzz:
        validator.fuzz_object(obj, file)
        return True
    else:
        return validator.validate_object(obj, file)


if __name__ == "__main__":
    valid = validate_sptx()

    if valid:
        sys.exit(0)
    else:
        sys.exit(1)
