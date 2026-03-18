qs_path := examples/quick_start
qs_py_files := $(wildcard $(qs_path)/*.py)
qs_py_run_targets := $(patsubst $(qs_path)/%,%,$(qs_py_files))

PYTHON := ipython

run-quick-start: $(qs_py_run_targets)

$(qs_py_run_targets): % :
	[ -e $(qs_path)/$*.skip ] || $(PYTHON) $(qs_path)/$*
