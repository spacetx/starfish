py_files := $(wildcard notebooks/*.py)
ipynb_files := $(wildcard notebooks/*.ipynb)
ipynb_validate_targets := $(addprefix validate__, $(ipynb_files))
ipynb_regenerate_targets := $(addprefix regenerate__, $(addsuffix .ipynb, $(basename $(py_files))))
py_regenerate_targets := $(addprefix regenerate__, $(addsuffix .py, $(basename $(ipynb_files))))

test: $(ipynb_validate_targets)
validate_notebooks: $(ipynb_validate_targets)
regenerate_ipynb: $(ipynb_regenerate_targets)
regenerate_py: $(py_regenerate_targets)

$(ipynb_validate_targets): TEMPFILE := $(shell mktemp)
$(ipynb_validate_targets): validate__%.ipynb :
	nbencdec encode $*.ipynb $(TEMPFILE)
	diff -q $*.py $(TEMPFILE)

$(ipynb_regenerate_targets): regenerate__%.ipynb : %.py
	nbencdec decode $< $*.ipynb

$(py_regenerate_targets): regenerate__%.py : %.py
	nbencdec encode $*.ipynb $<
