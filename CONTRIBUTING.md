# Contributing to Starfish

If you're reading this section, you're probably interested in contributing to Starfish.  Welcome and thanks for your interest in contributing!

# How can I contribute?
Starfish is designed to specify pipeline recipes for image processing. To support this use, the library is composed as a series of `pipeline_component` modules.
The objects that sub-class `pipeline_component` spawn a command-line interface that should identically track the API of the python library.

A typical starfish run consists of running one or more image processing filter stages, and identifying features through either a spot- or pixel-based approach.
The identified features are then decoded into the genes that they correspond to by mapping the fluorescence channel (and optionally hybridization round) using a codebook.
Finally, the filtered data are segmented, identifying which cell each feature belongs to.

Implementing a new module of an existing `pipeline_component` is a simple process. For example, to add another image filtering step, one would:

1. Create a new python file `new_filter.py` in the `starfish/pipeline/filter/` directory.

2. Create a new python class `NewFilter(PipelineComponent)` in the `new_filter.py` class.

3. Implement all required methods from the base class.

4. For the command line arguments to mimic the API, the arguments passed to the CLI must exactly match the names of the parameters for `NewFilter.__init__`, after dashes are automatically converted by argparse.
For example, `--foo-bar` would convert to `foo_bar` and init must accept such an argument: `NewFilter.__init__(foo_bar, ..., **kwargs)`

5. `NewFilter.__init__()` must have a `**kwargs` parameter to accept arbitrary CLI args.

That's it! your pipeline component will automatically register and be available under `starfish filter` in the CLI.
If at any point something gets confusing, it should be possible to look at existing pipeline components of the same category for guidance on implementation.

## Reporting bugs
- Bugs can be contributed as issues in the starfish repository. Please check to make sure there is no existing issue that describes the problem you
have identified before adding your bug.

## Code contributions
- Don't break the build: pull requests are expected to pass all automated CI checks. You can run those checks locally by running `make all` in
starfish repository root.

- All Pull Request comments must be addressed, even after merge.

- All code must be reviewed by at least 1 other team member.

- All code must have typed parameters, and when possible, typed return calls (see [PEP484](https://www.python.org/dev/peps/pep-0484)).
We also encourage rational use of in-line variable annotation when the type of a newly defined object is not clear (see [PEP526](https://www.python.org/dev/peps/pep-0526/).

- All code must be documented according to [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style guidelines.

- Numpy provides an excellent [development workflow](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html) that we encourage you to follow when developing features for starfish!

## Notebook contributions
- All `.ipynb` files should have a corresponding `.py` file.  Use [nbencdec](https://github.com/ttung/nbencdec) to generate the corresponding `.py` file.
The `.py` files allow refactor commands in the codebase to find code in the `.py` files, which is an important to keep the notebooks working as starfish evolves.
