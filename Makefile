SHELL := /bin/bash

MODULES=starfish examples validate_sptx

all:	lint mypy test

lint:
	# TODO: (ttung) suppressing lint checks for separate high-churn PR.
	# flake8 $(MODULES)
	echo "LINT SUPPRESSED"

test:
	pytest -v -n 8 --cov starfish --cov validate_sptx

mypy:
	mypy --ignore-missing-imports $(MODULES)

include notebooks/subdir.mk
