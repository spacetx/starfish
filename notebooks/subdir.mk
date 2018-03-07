ipynb_files := $(wildcard notebooks/*.ipynb)
ipynb_validate_targets := $(addprefix validate__, $(ipynb_files))

test: $(ipynb_validate_targets)
validate_notebooks: $(ipynb_validate_targets)

$(ipynb_validate_targets): TEMPFILE := $(shell mktemp)
$(ipynb_validate_targets): validate__%.ipynb :
	nbencdec encode $*.ipynb $(TEMPFILE)
	diff -q $*.py $(TEMPFILE)
