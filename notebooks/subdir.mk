py_files := $(wildcard notebooks/py/*.py)
ipynb_files := $(wildcard notebooks/*.ipynb)
py_run_targets := $(addprefix run__, $(py_files))
py_check_targets := $(addprefix check__, $(py_files))
ipynb_validate_targets := $(addprefix validate__, $(ipynb_files))
ipynb_regenerate_targets := $(addprefix regenerate__notebooks/, $(addsuffix .ipynb, $(notdir $(basename $(py_files)))))
py_regenerate_targets := $(addprefix regenerate__notebooks/py/, $(addsuffix .py, $(notdir $(basename $(ipynb_files)))))
PYTHON := python

fast: $(ipynb_validate_targets)
run-notebooks: $(py_run_targets)
check-notebooks: $(py_check_targets)
validate-notebooks: $(ipynb_validate_targets)
regenerate-ipynb: $(ipynb_regenerate_targets)
regenerate-py: $(py_regenerate_targets)

$(py_run_targets): run__%.py :
	[ -e $*.py.skip ] || $(PYTHON) $*.py

$(py_check_targets): check__%.py :
	grep -q $*.py .travis.yml

$(ipynb_validate_targets): TEMPFILE := $(shell mktemp)
$(ipynb_validate_targets): validate__notebooks/%.ipynb :
	nbencdec encode notebooks/$*.ipynb $(TEMPFILE)
	diff -q <(cat notebooks/py/$*.py | egrep -v '^# EPY: stripped_notebook: ') <(cat $(TEMPFILE) | egrep -v '# EPY: stripped_notebook: ')

$(ipynb_regenerate_targets): regenerate__notebooks/%.ipynb : notebooks/py/%.py
	nbencdec decode notebooks/py/$*.py notebooks/$*.ipynb

$(py_regenerate_targets): regenerate__notebooks/py/%.py : notebooks/%.ipynb
	nbencdec encode notebooks/$*.ipynb notebooks/py/$*.py
