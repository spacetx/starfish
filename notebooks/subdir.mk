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
	diff -q <(cat $*.py | egrep -v '^# EPY: stripped_notebook: ') <(cat $(TEMPFILE) | egrep -v '# EPY: stripped_notebook: ')

$(ipynb_regenerate_targets): regenerate__%.ipynb : %.py
	nbencdec decode $*.py $*.ipynb

$(py_regenerate_targets): regenerate__%.py : %.ipynb
	nbencdec encode $*.ipynb $*.py
