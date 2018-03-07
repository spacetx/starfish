MODULES=starfish tests

lint:
	flake8 $(MODULES)

test_srcs := $(wildcard tests/test_*.py)

test: $(test_srcs) lint
	coverage combine
	rm -f .coverage.*

$(test_srcs): %.py :
	coverage run -p --source=starfish -m unittest $(subst /,.,$*)

.PHONY : $(test_srcs)

include notebooks/subdir.mk
