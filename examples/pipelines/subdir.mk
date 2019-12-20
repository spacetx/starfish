path := examples/pipelines
py_files := $(wildcard $(path)/*.py)
sh_files := $(wildcard $(path)/*.sh)
py_run_targets := $(patsubst $(path)/%,%,$(py_files))
sh_run_targets := $(patsubst $(path)/%,%,$(sh_files))

PYTHON := ipython

run-examples: $(py_run_targets) $(sh_run_targets)

$(py_run_targets): % :
	[ -e $(path)/$*.skip ] || $(PYTHON) $(path)/$*

$(sh_run_targets): % :
	[ -e $(path)/$*.skip ] || $(SHELL) $(path)/$*
