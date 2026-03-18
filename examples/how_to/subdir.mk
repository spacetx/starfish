how_to_path := examples/how_to
how_to_py_files := $(wildcard $(how_to_path)/*.py)
how_to_py_run_targets := $(patsubst $(how_to_path)/%,%,$(how_to_py_files))

PYTHON := ipython

run-how-to: $(how_to_py_run_targets)

$(how_to_py_run_targets): % :
	[ -e $(how_to_path)/$*.skip ] || $(PYTHON) $(how_to_path)/$*
