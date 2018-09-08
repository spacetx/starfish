SHELL := /bin/bash

MODULES=starfish examples validate_sptx

all:	lint mypy test

lint:
	flake8 $(MODULES)

test:
	pytest -v -n 8 --cov starfish --cov validate_sptx

mypy:
	mypy --ignore-missing-imports $(MODULES)

include notebooks/subdir.mk
