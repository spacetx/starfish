path := docs/source/_static/data_processing_examples
py_files := $(wildcard $(path)/*.py)
sh_files := $(wildcard $(path)/*.sh)
py_run_targets := $(patsubst $(path)/%,run__%,$(py_files))
sh_run_targets := $(patsubst $(path)/%,run__%,$(sh_files))

PYTHON := ipython

run-examples: $(py_run_targets) $(sh_run_targets)

$(py_run_targets): run__% :
	[ -e $(path)/$*.skip ] || $(PYTHON) $(path)/$*

$(sh_run_targets): run__% :
	[ -e $(path)/$*.skip ] || $(SHELL) $(path)/$*
