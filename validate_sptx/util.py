import json
import os
import warnings
from pkg_resources import resource_filename
from typing import Dict, Iterator

from jsonschema import RefResolver, Draft4Validator, ValidationError


class SpaceTxValidator:

    def __init__(self, schema: str, fuzz: bool=False) -> None:
        """create a validator for a json-schema compliant spaceTx specification file

        Parameters
        ----------
        schema : str
            file path to schema
        fuzz : bool
            if true, then the json documents which are validated will
            be modified piece-wise and a statement printed to standard
            out about whether or no they are still valid.

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
        experiment_schema_path = resource_filename("validate_sptx", "schema/experiment.json")
        package_root = os.path.dirname(os.path.dirname(experiment_schema_path))
        base_uri = 'file://' + package_root + '/'
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
        filename : str
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
                stars="***" * level,  level=str(level), path="/".join(error.absolute_schema_path),
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

        """
        target_object = self.load_json(target_file)
        return self.validate_object(target_object, target_file)

    def validate_object(self, target_object: Dict, target_file: str=None) -> bool:
        """validate a loaded json object, returning True if valid, and False otherwise

        Parameters
        ----------
        target_object : Dict
            loaded json object to be validated against the schem passed to this object's constructor
        target_file : str
            informational string regarding the source file of the given object

        Returns
        -------
        bool :
            True, if object valid, else False

        """
        if self._validator.is_valid(target_object):
            return True
        else:
            es: Iterator[ValidationError] = self._validator.iter_errors(target_object)
            self._recurse_through_errors(es, filename=target_file)
            return False
