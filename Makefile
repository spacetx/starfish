SHELL := /bin/bash

MODULES=starfish tests

all:	lint mypy test

lint:
	flake8 $(MODULES)

mypy:
	mypy --ignore-missing-imports $(MODULES)

test_srcs := $(wildcard tests/test_*.py)

test: STARFISH_COVERAGE := 1
test: $(test_srcs) lint
	coverage combine
	rm -f .coverage.*

$(test_srcs): %.py :
	if [ "$(STARFISH_COVERAGE)" == 1 ]; then \
		STARFISH_COVERAGE=1 coverage run -p --source=starfish -m unittest $(subst /,.,$*); \
	else \
		python -m unittest $(subst /,.,$*); \
	fi

.PHONY : $(test_srcs)

include notebooks/subdir.mk
