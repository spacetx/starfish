SHELL := /bin/bash

MODULES=validate_sptx

all:	lint mypy test

lint:
	flake8 $(MODULES)

test:
	pytest -v

mypy:
	mypy --ignore-missing-imports $(MODULES)
