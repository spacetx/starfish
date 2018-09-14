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

refresh_all_requirements:
	@echo -n '' >| REQUIREMENTS.txt
	@echo -n '' >| REQUIREMENTS-DEV.txt
	@if [ $$(uname -s) == "Darwin" ]; then sleep 1; fi  # this is require because Darwin HFS+ only has second-resolution for timestamps.
	@touch REQUIREMENTS.txt.in REQUIREMENTS-DEV.txt.in
	@$(MAKE) REQUIREMENTS.txt REQUIREMENTS-DEV.txt

REQUIREMENTS.txt REQUIREMENTS-DEV.txt : %.txt : %.txt.in
	[ ! -e .requirements-env ] || exit 1
	python -m venv .$<-env
	.$<-env/bin/pip install -r $@
	.$<-env/bin/pip install -r $<
	echo "# You should not edit this file directly.  Instead, you should edit one of the following files ($^) and run make $@" >| $@
	.$<-env/bin/pip freeze >> $@
	rm -rf .$<-env

REQUIREMENTS-DEV.txt : REQUIREMENTS.txt.in

include notebooks/subdir.mk

.PHONY: all lint lint-non-init lint-init test mypy refresh_all_requirements
