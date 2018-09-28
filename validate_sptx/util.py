import copy
import json
import os
import sys
import warnings
from typing import Any, Dict, IO, Iterator, List, Optional, Union

from jsonschema import Draft4Validator, RefResolver, ValidationError
from pkg_resources import resource_filename


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
                stars="***" * level, level=str(level), path="/".join(error.absolute_schema_path),
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
        self.stack: List[Any] = []
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
