SHELL := /bin/bash

MODULES=starfish examples validate_sptx

all:	lint mypy test

lint:   lint-non-init lint-init

lint-non-init:
	flake8 --ignore 'E301, E302, E305, E401, E731, F811' --exclude='*__init__.py' $(MODULES)

lint-init:
	flake8 --ignore 'E301, E302, E305, E401, F401, E731, F811' --filename='*__init__.py' $(MODULES)

test:
	pytest -v -n 8 --cov starfish --cov validate_sptx

mypy:
	mypy --ignore-missing-imports $(MODULES)

docs-%:
	make -C docs $*

include notebooks/subdir.mk

.PHONY: all lint lint-non-init lint-init test mypy
