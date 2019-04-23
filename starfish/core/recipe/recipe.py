from typing import (
    AbstractSet,
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
)

import numpy as np

from starfish import image, spots
from starfish.types import Axes, Coordinates
from .filesystem import FileProvider, FileTypes
from .runnable import Runnable


class ExecutionComplete(Exception):
    """Raised by :py:class:`Execution` when it is complete.  We don't rely on catching StopIteration
    because some underlying library may have raised that instead."""
    pass


class _Execution:
    """Encompasses the state of a single execution of a recipe."""
    def __init__(
            self,
            runnable_sequence: Sequence[Runnable],
            output_runnables: Sequence[Runnable],
            output_paths: Sequence[str],
    ) -> None:
        self._runnable_sequence = iter(runnable_sequence)
        self._output_runnables = output_runnables
        self._output_paths = output_paths

        # build a map between each runnable and its dependents.  each time a runnable completes, we
        # go through each of its dependencies to see if its results are still needed.
        runnable_dependents: MutableMapping[Runnable, Set[Runnable]] = dict()
        for output_runnable in self._output_runnables:
            _Execution._build_graph(output_runnable, runnable_dependents)
        self.runnable_dependents: Mapping[Runnable, AbstractSet[Runnable]] = runnable_dependents

        # completed results
        self._completed_runnables: Set[Runnable] = set()
        self._completed_results: MutableMapping[Runnable, Any] = dict()

    def _run_one_tick(self) -> None:
        """Run one tick of the execution graph.  Raises StopIteration if it's done."""
        try:
            runnable = next(self._runnable_sequence)
        except StopIteration as ex:
            raise ExecutionComplete from ex

        result = runnable.run(self._completed_results)

        # record what's been done.
        self._completed_runnables.add(runnable)
        self._completed_results[runnable] = result

        # examine all the dependencies, and discard the results if no one else needs it.
        for dependency in runnable.runnable_dependencies:
            if dependency in self._output_runnables:
                # it's required by the outputs, so preserve this.
                continue

            for dependent in self.runnable_dependents[dependency]:
                if dependent not in self._completed_runnables:
                    # someone still needs this runnable's result.
                    break
            else:
                # every dependent is complete.  drop the result.
                del self._completed_results[dependency]

    def _save(self) -> None:
        for runnable, output_path in zip(self._output_runnables, self._output_paths):
            # get the result
            result = self._completed_results[runnable]

            filetype = FileTypes.resolve_by_instance(result)
            filetype.save(result, output_path)

    def run_and_save(self) -> None:
        """Run execution graph to completion and save the results to disk."""
        while True:
            try:
                self._run_one_tick()
            except ExecutionComplete:
                break

        self._save()

    @staticmethod
    def _build_graph(
            runnable: Runnable,
            runnable_dependents: MutableMapping[Runnable, Set[Runnable]],
            seen_runnables: Optional[Set[Runnable]]=None,
    ) -> None:
        if seen_runnables is None:
            seen_runnables = set()

        if runnable in seen_runnables:
            return
        seen_runnables.add(runnable)

        for dependency in runnable.runnable_dependencies:
            # mark ourselves a dependent of each of our dependencies.
            if dependency not in runnable_dependents:
                runnable_dependents[dependency] = set()
            runnable_dependents[dependency].add(runnable)
            _Execution._build_graph(dependency, runnable_dependents, seen_runnables)


class OrderedSequence:
    def __init__(self) -> None:
        self._sequence: MutableSequence[Runnable] = list()

    def __call__(self, *args, **kwargs):
        result = Runnable(*args, **kwargs)
        self._sequence.append(result)
        return result

    @property
    def sequence(self) -> Sequence[Runnable]:
        return self._sequence


class Recipe:
    """Recipes are snippets of python code that describe an image processing pipeline.  Executing a
    recipe requires pairing the recipe code with locations of input files and the locations to write
    outputs to.

    A few additional variables are injected into the recipe's scope:

    ====================  ==========================================================================
     Variable name         Description
    --------------------  --------------------------------------------------------------------------
     file_inputs           A 0-based array with the locations of input files provided for a recipe's
                           execution.
     file_outputs          Any objects written to this 0-based array will be written out to disk at
                           the end of a recipe's execution.
     np                    A reference to the numpy library.
     Axes                  A reference to the :py:class:`~starfish.types.Axes` enum.
     Coordinates           A reference to the :py:class:`~starfish.types.Coordinates` enum.
     ApplyTransform        A reference to :py:class:`~starfish.image.ApplyTransform`.
     Filter                A reference to :py:class:`~starfish.image.Filter`.
     LearnTransform        A reference to :py:class:`~starfish.image.LearnTransform`.
     Segment               A reference to :py:class:`~starfish.image.Segment`.
     Decode                A reference to :py:class:`~starfish.spots.Decode`.
     DetectPixels          A reference to :py:class:`~starfish.spots.DetectPixels`.
     DetectSpots           A reference to :py:class:`~starfish.spots.DetectSpots`.
     AssignTargets         A reference to :py:class:`~starfish.spots.AssignTargets`.
    ====================  ==========================================================================

    Recipes also have access to a ``compute`` method:
    :code:`compute(algorithm_class, parameters_to_run_method...,
    constructor_argument_name=constructor_argument_value....)`

    The first parameter is the algorithm class that implements a pipeline algorithm.  It should
    subclass :py:class:`~starfish.pipeline.AlgorithmBase`.

    The subsequent positional parameters are the parameters passed to the ``run()`` method of the
    algorithm class.

    The keyword parameters (parameters where a name and a value are provided) are the parameters
    passed to the algorithm class's constructor.

    Parameters can include input files that are passed in as a member of the ``file_inputs`` array.
    It can also include the output of previous ``compute()`` method calls.

    The return value of ``compute()`` method calls is a future, i.e., a promise for the data.  It
    should not be assumed that the resulting data will be available during the recipe's scope.
    """
    def __init__(
            self,
            recipe_str: str,
            input_paths_or_urls: Sequence[str],
            output_paths: Sequence[str],
    ):
        ordered_sequence = OrderedSequence()
        file_outputs: MutableMapping[int, Runnable] = {}
        recipe_scope = {
            "file_inputs": [
                FileProvider(input_path_or_url)
                for input_path_or_url in input_paths_or_urls
            ],
            "compute": ordered_sequence,
            "file_outputs": file_outputs,
        }

        # inject numpy.
        recipe_scope['np'] = np

        component: Any

        # inject some useful starfish classes.
        for component in (Axes, Coordinates):
            recipe_scope[component.__name__] = component

        # inject all the pipeline components.
        for component in (
                image.ApplyTransform, image.Filter, image.LearnTransform, image.Segment):
            recipe_scope[component.__name__] = component
        for component in (
                spots.Decode, spots.DetectPixels, spots.DetectSpots, spots.AssignTargets):
            recipe_scope[component.__name__] = component

        ast = compile(recipe_str, "<string>", "exec")
        exec(ast, recipe_scope)

        assert len(file_outputs) == len(output_paths), \
            "Recipe generates more outputs than output paths provided!"

        # verify that the outputs are sequential.
        ordered_outputs: MutableSequence[Runnable] = list()
        for ix in range(len(file_outputs)):
            assert ix in file_outputs, \
                f"file_outputs[{ix}] is not set"
            assert isinstance(file_outputs[ix], Runnable), \
                f"file_outputs[{ix}] is not the result of a compute(..)"
            ordered_outputs.append(file_outputs[ix])

        self._runnable_order = ordered_sequence.sequence
        self._outputs: Sequence[Runnable] = ordered_outputs
        self._output_paths = output_paths

    def _execution(self) -> _Execution:
        return _Execution(self._runnable_order, self._outputs, self._output_paths)

    def run_and_save(self):
        """Run recipe to completion and save the results to disk."""
        execution = self._execution()
        execution.run_and_save()
