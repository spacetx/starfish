import inspect
import warnings
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
    Type,
)

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent, PipelineComponentType
from .errors import (
    ConstructorError,
    ConstructorExtraParameterWarning,
    ExecutionError,
    RunInsufficientParametersError,
    TypeInferenceError,
)
from .filesystem import FileProvider, TypedFileProvider


class Runnable:
    """Runnable represents a single invocation of a pipeline component, with a specific algorithm
    implementation.  For arguments to the algorithm's constructor and run method, it can accept
    :py:class:`starfish.recipe.filesystem.FileProvider` objects, which represent a file path or url.
    For arguments to the algorithm's run method, it can accept the results of other Runnables.

    One can compose any starfish pipeline run using a directed acyclic graph of Runnables objects.
    """
    def __init__(
            self,
            pipeline_component_name: str,
            algorithm_name: str,
            *inputs,
            **algorithm_options
    ) -> None:
        self._pipeline_component_name = pipeline_component_name
        self._algorithm_name = algorithm_name
        self._raw_inputs = inputs
        self._raw_algorithm_options = algorithm_options

        self._pipeline_component_cls: Type[PipelineComponent] = \
            PipelineComponentType.get_pipeline_component_type_by_name(self._pipeline_component_name)
        self._algorithm_cls: Type = getattr(
            self._pipeline_component_cls, self._algorithm_name)

        # retrieve the actual __init__ method
        signature = Runnable._get_actual_method_signature(
            self._algorithm_cls.__init__)
        formatted_algorithm_options = self._format_algorithm_constructor_arguments(
            signature, self._raw_algorithm_options, self.__str__)

        try:
            self._algorithm_instance: AlgorithmBase = self._algorithm_cls(
                **formatted_algorithm_options)
        except Exception as ex:
            raise ConstructorError(f"Error instantiating the algorithm for {str(self)}") from ex

        # retrieve the actual run method
        signature = Runnable._get_actual_method_signature(
            self._algorithm_instance.run)  # type: ignore
        self._inputs = self._format_run_arguments(signature, self._raw_inputs, self.__str__)

    @staticmethod
    def _get_actual_method_signature(run_method: Callable) -> inspect.Signature:
        if hasattr(run_method, "__closure__"):
            # it's a closure, probably because of AlgorithmBaseType.run_with_logging.  Unwrap to
            # find the underlying method.
            closure = run_method.__closure__  # type: ignore
            if closure is not None:
                run_method = closure[0].cell_contents

        return inspect.signature(run_method)

    @staticmethod
    def _format_algorithm_constructor_arguments(
            constructor_signature: inspect.Signature,
            algorithm_options: Mapping[str, Any],
            str_callable: Callable[[], str],
    ) -> Mapping[str, Any]:
        """Given the constructor's signature and a mapping of keyword argument names to values,
        format them such that the constructor can be invoked.

        Some parameters may be :py:class:`starfish.recipe.filesystem.FileProvider` instances.  Use
        the type hints in the constructor's signature to identify the expected file type, and load
        them into memory accordingly.

        Parameters
        ----------
        constructor_signature : inspect.Signature
            The signature for the constructor.
        algorithm_options : Mapping[str, Any]
            The parameters for the constructor, as provided to the Runnable.
        str_callable : Callable[[], str]
            A callable that can be invoked to provide a user-friendly representation of the
            Runnable, in case any errors or warnings are generated.

        Returns
        -------
        Mapping[str, Any] : The parameters for the constructor, ready to be passed into the
        constructor.
        """
        parameters = constructor_signature.parameters
        assert next(iter(parameters.keys())) == "self"

        formatted_algorithm_options: MutableMapping[str, Any] = {}
        for algorithm_option_name, algorithm_option_value in algorithm_options.items():
            if isinstance(algorithm_option_value, Runnable):
                raise RuntimeError("Runnable's constructors cannot depend on another runnable")

            try:
                option_class = parameters[algorithm_option_name].annotation
            except KeyError:
                warnings.warn(
                    f"Constructor for {str_callable()} does not have an explicitly typed "
                    + f"parameter {algorithm_option_name}.",
                    category=ConstructorExtraParameterWarning,
                )
                continue

            if isinstance(algorithm_option_value, FileProvider):
                try:
                    provider = TypedFileProvider(algorithm_option_value, option_class)
                except TypeError as ex:
                    raise TypeInferenceError(
                        f"Error inferring the types for the parameters to the algorithm's"
                        + f" constructor for {str_callable()}") from ex
                formatted_algorithm_options[algorithm_option_name] = provider.load()
            else:
                formatted_algorithm_options[algorithm_option_name] = algorithm_option_value

        return formatted_algorithm_options

    @staticmethod
    def _format_run_arguments(
            run_signature: inspect.Signature,
            inputs: Sequence,
            str_callable: Callable[[], str],
    ) -> Sequence:
        """Given the run method's signature and a sequence of parameters, format them such that the
        run method can be invoked.

        Some parameters may be :py:class:`starfish.recipe.filesystem.FileProvider` instances.  Use
        the type hints in the run method's signature to identify the expected file type, and load
        them into memory accordingly.

        Parameters that are the outputs of other Runnables are not resolved to their values until
        the run method is invoked.  Therefore, the sequence of parameters returned may include
        the dependent Runnable objects.

        Parameters
        ----------
        run_signature : inspect.Signature
            The signature for the run method.
        inputs : Sequence
            The parameters for the run method, as provided to the Runnable.
        str_callable : Callable[[], str]
            A callable that can be invoked to provide a user-friendly representation of the
            Runnable, in case any errors or warnings are generated.

        Returns
        -------
        Sequence : The parameters for the run method, ready to be passed into the constructor,
                   except for dependent Runnables, which are resolved later.
        """
        formatted_inputs: MutableSequence = []

        keys_iter = iter(run_signature.parameters.keys())
        inputs_iter = iter(inputs)

        # first parameter to the run method should be "self"
        assert next(keys_iter) == "self"

        # match up the parameters as best as we can.
        for _input, key in zip(inputs_iter, keys_iter):
            if isinstance(_input, FileProvider):
                annotation = run_signature.parameters[key].annotation
                try:
                    provider = TypedFileProvider(_input, annotation)
                except TypeError as ex:
                    raise TypeInferenceError(
                        f"Error inferring the types for the parameters to the algorithm's"
                        + f" run method for {str_callable()}") from ex
                formatted_inputs.append(provider)
            else:
                formatted_inputs.append(_input)

        # are there any parameters left in the signature?  if so, they must have default values
        # because we don't have values.
        no_default = inspect._empty  # type: ignore

        for key in keys_iter:
            if (run_signature.parameters[key].default == no_default
                    and run_signature.parameters[key].kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
            )):
                raise RunInsufficientParametersError(f"No value for parameter {key}")

        return formatted_inputs

    @property
    def runnable_dependencies(self) -> Set["Runnable"]:
        """Retrieves a set of Runnables that this Runnable depends on."""
        return set(runnable for runnable in self._inputs if isinstance(runnable, Runnable))

    def run(self, previous_results: Mapping["Runnable", Any]) -> Any:
        """Invoke the run method.  Results for dependent Runnables are retrieved from the
        `previous_results` mapping.

        Parameters
        ----------
        previous_results : Mapping[Runnable, Any]
            The results calculated thus far in an execution run.

        Returns
        -------
        The result from invoking the run method.
        """
        inputs = list()
        for _input in self._inputs:
            if isinstance(_input, Runnable):
                inputs.append(previous_results[_input])
            elif isinstance(_input, TypedFileProvider):
                inputs.append(_input.load())
            else:
                inputs.append(_input)
        try:
            return self._algorithm_instance.run(*inputs)  # type: ignore
        except Exception as ex:
            raise ExecutionError(f"Error running the algorithm for {str(self)}") from ex

    def __str__(self):
        inputs_arr = [""]
        inputs_arr.extend([str(raw_input) for raw_input in self._raw_inputs])
        algorithm_options_arr = [""]
        algorithm_options_arr.extend([
            f"{algorithm_option_name}={str(algorithm_option_value)}"
            for algorithm_option_name, algorithm_option_value in
            self._raw_algorithm_options.items()
        ])

        inputs_str = ", ".join(inputs_arr)
        algorithm_options_str = ", ".join(algorithm_options_arr)

        return (f"compute("
                + f"\"{self._pipeline_component_name}\","
                + f" \"{self._algorithm_name}\""
                + f"{inputs_str}"
                + f"{algorithm_options_str})")
