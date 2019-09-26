import copy
import json
import posixpath
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, IO, Iterator, List, Optional, Union

from jsonschema import Draft4Validator, RefResolver, ValidationError
from pkg_resources import resource_filename
from semantic_version import Version
from slicedimage import VERSIONS as SLICEDIMAGE_VERSIONS

from starfish.core.codebook._format import CURRENT_VERSION as CODEBOOK_CURRENT_VERSION
from starfish.core.experiment.version import CURRENT_VERSION as EXPERIMENT_CURRENT_VERSION


TILESET_CURRENT_VERSION = Version(SLICEDIMAGE_VERSIONS[-1].VERSION)
package_name = 'starfish'


def _get_absolute_schema_path(schema_name: str) -> str:
    """turn the name of the schema into an absolute path by joining it to <package_root>/schema."""
    return resource_filename("starfish", posixpath.join("spacetx_format", "schema", schema_name))


class SpaceTxValidator:

    def __init__(self, schema: str) -> None:
        """create a validator for a json-schema compliant spaceTx specification file

        Parameters
        ----------
        schema : str
            file path to schema

        """
        self._schema: Dict = self.load_json(schema)
        self._validator: Draft4Validator = self._create_validator(self._schema)

    @staticmethod
    def _create_validator(schema: Dict) -> Draft4Validator:
        """resolve $ref links in a loaded json schema and return a validator

        Parameters
        ----------
        schema : Dict
            loaded json schema

        Returns
        -------
        Draft4Validator :
            json-schema validator specific to the supplied schema, with references resolved

        """

        # Note: we are using 5.0.0 here as the first known file. It does *not* need to
        # be upgraded with each version bump since only the dirname is used.
        experiment_schema_path = Path(resource_filename(
            package_name, "spacetx_format/schema/experiment_5.0.0.json"))

        package_root_path = experiment_schema_path.parent.parent
        base_uri = f"{package_root_path.as_uri()}/"
        resolver = RefResolver(base_uri, schema)
        return Draft4Validator(schema, resolver=resolver)

    @staticmethod
    def load_json(json_file: str) -> Dict:
        with open(json_file, 'rb') as f:
            return json.load(f)

    @staticmethod
    def _recurse_through_errors(error_iterator: Iterator[ValidationError],
                                level: int=0,
                                filename: str=None) -> None:
        """Recurse through ValidationErrors, printing message and schema path

        Parameters
        ----------
        error_iterator : Iterator[ValidationError]
            iterator over ValidationErrors that occur during validation
        level : int
            current level of recursion
        filename: str
            informational string regarding the source file of the given object

        """
        fmt = "\n{stars} {message}\n"
        fmt += "\tSchema:         \t{schema}\n"
        fmt += "\tSubschema level:\t{level}\n"
        fmt += "\tPath to error:  \t{path}\n"
        if filename:
            fmt += "\tFilename:       \t{filename}\n"
        for error in error_iterator:
            message = fmt.format(
                stars="***" * level, level=str(level),
                path="/".join([str(x)for x in error.absolute_schema_path]),
                message=error.message, cause=error.cause, schema=error.schema.get("$id", "unknown"),
                filename=filename,
            )
            warnings.warn(message)
            if error.context:
                level += 1
                SpaceTxValidator._recurse_through_errors(error.context, level=level)

    def validate_file(self, target_file: str) -> bool:
        """validate a target file, returning True if valid and False otherwise

        Parameters
        ----------
        target_file : str
            path or URL to a target json object to be validated against the schema passed to this
            object's constructor

        Returns
        -------
        bool :
            True, if object valid, else False

        Examples
        --------
        Validate a codebook file::

            >>> from pkg_resources import resource_filename
            >>> from starfish.core.spacetx_format.util import SpaceTxValidator
            >>> schema_path = resource_filename(
                    "starfish", "spacetx_format/schema/codebook/codebook.json")
            >>> validator = SpaceTxValidator(schema_path)
            >>> if not validator.validate_file(your_codebook_filename):
            >>>     raise Exception("invalid")

        """
        target_object = self.load_json(target_file)
        return self.validate_object(target_object, target_file)

    def validate_object(
            self,
            target_object: Union[dict, list],
            target_file: str=None,
    ) -> bool:
        """validate a loaded json object, returning True if valid, and False otherwise

        Parameters
        ----------
        target_object : Dict
            loaded json object to be validated against the schema passed to this object's
            constructor
        target_file : str
            informational string regarding the source file of the given object

        Returns
        -------
        bool :
            True, if object valid, else False

        Examples
        --------
        Validate an experiment json string ::

            >>> from pkg_resources import resource_filename
            >>> from starfish.core.spacetx_format.util import SpaceTxValidator
            >>> schema_path = resource_filename("starfish", "spacetx_format/schema/experiment.json")
            >>> validator = SpaceTxValidator(schema_path)
            >>> if not validator.validate_object(your_experiment_object):
            >>>     raise Exception("invalid")

        """

        if self._validator.is_valid(target_object):
            return True
        else:
            es: Iterator[ValidationError] = self._validator.iter_errors(target_object)
            self._recurse_through_errors(es, filename=target_file)
            return False

    def fuzz_object(
            self,
            target_object: Union[dict, list],
            target_file: str=None,
            out: IO=sys.stdout,
    ) -> None:
        """performs mutations on the given object and tests for validity.

        A representation of the validity is printed to the given output stream.

        Parameters
        ----------
        target_object : Dict
            loaded json object to be validated against the schema passed to this object's
            constructor
        target_file : str
            informational string regarding the source file of the given object
        out : IO
            output stream for printing

        """

        if target_file:
            out.write(f"> Fuzzing {target_file}...\n")
        else:
            out.write("> Fuzzing unknown...\n")
        fuzzer = Fuzzer(self._validator, target_object, out)
        fuzzer.fuzz()


class Fuzzer(object):

    def __init__(self, validator: Draft4Validator, obj: Any, out: IO=sys.stdout) -> None:
        """create a fuzzer which will check different situations against the validator

        Parameters
        ----------
        validator : SpaceTxValidator
            validator which should match the given object type
        obj : Any
            JSON-like object which will be checked against the validator
        out : IO
            output stream for printing

        """
        self.validator = validator
        self.obj = obj
        self.out = out
        self.stack: Optional[List[Any]] = None

    def fuzz(self) -> None:
        """prints to the out field the state of the object tree after types of fuzzing

        Each line is prefixed by the output of {state()} followed by a YAML-like
        representation of the branch of the object tree.
        """
        header = f"{self.state()}"
        header += "If the letter is present, mutation is valid!"
        self.out.write(f"{header}\n")
        self.out.write("".join([x in ("\t", "\n") and x or "-" for x in header]))
        self.out.write("\n")
        self.stack = []
        try:
            self.descend(self.obj)
        finally:
            self.stack = None

    def state(self) -> str:
        """primary driver for the checks of individual trees

        Returns
        -------
        str :
            space-separated representation of the fuzzing conditions.
            If a letter is present, then mutation leaves the tree in
            a valid state:

             A: inserting a fake key or appending to a list
             D: deleting a key or index
             I: converting value to an integer
             I: converting value to a string
             M: converting value to an empty dict
             L: converting value to an empty list

        """
        rv = [
            Add().check(self),
            Del().check(self),
            Change("I", lambda *args: 123456789).check(self),
            Change("S", lambda *args: "fake").check(self),
            Change("M", lambda *args: dict()).check(self),
            Change("L", lambda *args: list()).check(self),
        ]
        return ' '.join(rv) + "\t"

    def descend(self, obj: Any, depth: int=0, prefix: str="") -> None:
        """walk a JSON-like object tree printing the state of the tree
        at each level. A YAML representation is used for simplicity.

        Parameters
        ----------
        obj : Any
            JSON-like object tree
        depth : int
            depth in the tree that is currently being evaluated
        prefix : str
            value which should be prepended to printouts at this level
        """
        if self.stack is None:
            raise Exception("invalid state")
        if isinstance(obj, list):
            for i, o in enumerate(obj):
                depth += 1
                self.stack.append(i)
                self.descend(o, depth, prefix="- ")
                self.stack.pop()
                depth -= 1
        elif isinstance(obj, dict):
            for k in obj:
                # This is something of a workaround in that we need a special
                # case for object keys since no __getitem__ method will suffice.
                self.stack.append((k,))
                self.out.write(f"{self.state()}{' ' * depth}{prefix}{k}:\n")
                self.stack.pop()
                if prefix == "- ":
                    prefix = "  "
                depth += 1
                self.stack.append(k)
                self.descend(obj[k], depth, prefix="  " + prefix)
                self.stack.pop()
                depth -= 1
        else:
            self.out.write(f"{self.state()}{' ' * depth}{prefix}{obj}\n")

class Checker(object):

    @property
    def LETTER(self) -> str:
        return "?"

    def check(self, fuzz: Fuzzer) -> str:
        """create a copy of the current state of the object tree,
        mutate it, and run it through is_valid on the validator.

        Parameters
        ----------
        fuzz : Fuzzer
            the containing instance

        Returns
        -------
        str :
            A single character string representation of the check

        """
        # Don't mess with the top level
        if fuzz.stack is None:
            return self.LETTER
        if not fuzz.stack:
            return "-"
        # Operate on a copy for mutating
        dupe = copy.deepcopy(fuzz.obj)
        target = dupe
        for level in fuzz.stack[0:-1]:
            target = target.__getitem__(level)
        self.handle(fuzz, target)
        valid = fuzz.validator.is_valid(dupe)
        return valid and self.LETTER or "."

    def handle(self, fuzz, target):
        raise NotImplementedError()

class Add(Checker):

    @property
    def LETTER(self) -> str:
        return "A"

    def handle(self, fuzz, target):
        if isinstance(target, dict):
            target["fake"] = "!"
        elif isinstance(target, list):
            target.append("!")
        else:
            raise Exception("unknown")

class Del(Checker):

    @property
    def LETTER(self) -> str:
        return "D"

    def handle(self, fuzz, target):
        key = fuzz.stack[-1]
        if isinstance(key, tuple):
            key = key[0]
        target.__delitem__(key)

class Change(Checker):

    @property
    def LETTER(self) -> str:
        return self.letter

    def __init__(self, letter, call):
        self.letter = letter
        self.call = call

    def handle(self, fuzz, target):
        key = fuzz.stack[-1]
        if isinstance(key, tuple):
            key = key[0]
        target.__setitem__(key, self.call())


def get_schema_path(
        schema: str,
        doc: Optional[dict] = None,
        version: Optional[Version] = None,
) -> str:
    """lookup the absolute schema path, including version, based
    on the given parameters

    Parameters
    ----------
    schema : str
        A portion of the schema path. The heuristic applied is that if any
        of the strings "codebook", "experiment", "fov_manifest", "coordinates",
        "indices", "tiles", or "field_of_view" is found in the string, then the
        appropriate schema is returned. Otherwise an exception is raised.
    doc: dict
        If provided, the "version" key will be read from it.
    version: Version
        Default version string to use if not found in the doc.

    Returns
    -------
    str :
        absolute schema path, never None.

    """

    if doc is not None:
        version_str = doc.get("version", version)
        if version_str is not None:
            version = Version(version_str)

    if version is None:
        raise ValueError("Could not find the version of the schema to validate against")

    if "codebook" in schema:
        path = f"codebook_{version}/codebook.json"
    elif "experiment" in schema:
        path = f"experiment_{version}.json"
    elif "fov_manifest" in schema:
        path = f"fov_manifest_{version}.json"
    elif "coordinates" in schema:
        path = f"field_of_view_{version}/tiles/coordinates.json"
    elif "indices" in schema:
        path = f"field_of_view_{version}/tiles/indices.json"
    elif "tiles" in schema:
        path = f"field_of_view_{version}/tiles/tiles.json"
    elif "field_of_view" in schema:
        path = f"field_of_view_{version}/field_of_view.json"
    else:
        raise Exception(f"Unknown schema: {schema}")
    return _get_absolute_schema_path(path)


class CodebookValidator(SpaceTxValidator):
    """
    Subclass of SpaceTxValidator which enforces the use of the "codebook" schema
    as returned by get_schema_path.
    """

    def __init__(self, doc: Optional[dict] = None, version: Optional[Version] = None):
        super(CodebookValidator, self).__init__(get_schema_path("codebook", doc, version))


class ExperimentValidator(SpaceTxValidator):
    """
    Subclass of SpaceTxValidator which enforces the use of the "experiment" schema
    as returned by get_schema_path.
    """

    def __init__(self, doc: Optional[dict] = None, version: Optional[Version] = None):
        super(ExperimentValidator, self).__init__(get_schema_path("experiment", doc, version))


class FOVValidator(SpaceTxValidator):
    """
    Subclass of SpaceTxValidator which enforces the use of the "field_of_view" schema
    as returned by get_schema_path.
    """

    def __init__(self, doc: Optional[dict] = None, version: Optional[Version] = None):
        super(FOVValidator, self).__init__(get_schema_path("field_of_view", doc, version))


class ManifestValidator(SpaceTxValidator):
    """
    Subclass of SpaceTxValidator which enforces the use of the "fov_manifest" schema
    as returned by get_schema_path.
    """

    def __init__(self, doc: Optional[dict] = None, version: Optional[Version] = None):
        super(ManifestValidator, self).__init__(get_schema_path("fov_manifest", doc, version))


LatestCodebookValidator = lambda: CodebookValidator(version=CODEBOOK_CURRENT_VERSION)


LatestExperimentValidator = lambda: ExperimentValidator(version=EXPERIMENT_CURRENT_VERSION)


LatestFOVValidator = lambda: FOVValidator(version=TILESET_CURRENT_VERSION)


LatestManifestValidator = lambda: ManifestValidator(version=TILESET_CURRENT_VERSION)
