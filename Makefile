SHELL := /bin/bash

MPLBACKEND?=Agg
export MPLBACKEND

MODULES=starfish data_formatting_examples

define print_help
    @printf "    %-24s   $(2)\n" $(1)
endef

define create_venv
    if [[ "$(TRAVIS)" = "" ]]; then \
        python -m venv $(1); \
    else \
        virtualenv -p $$(which python) $(1); \
    fi
endef


all:	fast

### UNIT #####################################################
#
fast:	lint mypy fast-test docs-html

lint:   lint-non-init lint-init

lint-non-init:
	flake8 --ignore 'E252, E301, E302, E305, E401, W503, E731, F811' --exclude='*__init__.py' $(MODULES)

lint-init:
	flake8 --ignore 'E252, E301, E302, E305, E401, F401, W503, E731, F811' --filename='*__init__.py' $(MODULES)

test:
	pytest -v -n 8 --cov starfish

fast-test:
	pytest -v -n 8 --cov starfish -m 'not slow'

slow-test:
	pytest -v -n 8 --cov starfish -m 'slow'

mypy:
	mypy --ignore-missing-imports $(MODULES)

help-unit:
	$(call print_help, all, alias for fast)

.PHONY: all fast lint lint-non-init lint-init test mypy help-unit
#
##############################################################

### DOCS #####################################################
#
docs-%:
	make -C docs $*

help-docs:
	$(call print_help, docs-TASK, alias for 'make TASK' in the docs subdirectory)

.PHONY: help-docs
#
##############################################################

### REQUIREMENTS #############################################
#
check-requirements:
	if [[ $$(git status --porcelain REQUIREMENTS*) ]]; then \
	    echo "Modifications found in REQUIREMENTS files"; exit 2; \
	fi

refresh-all-requirements:
	@echo -n '' >| REQUIREMENTS-STRICT.txt
	@if [ $$(uname -s) == "Darwin" ]; then sleep 1; fi  # this is require because Darwin HFS+ only has second-resolution for timestamps.
	@touch REQUIREMENTS.txt
	@$(MAKE) REQUIREMENTS-STRICT.txt

REQUIREMENTS-STRICT.txt : REQUIREMENTS.txt
	[ ! -e .$<-env ] || exit 1
	$(call create_venv, .$<-env)
	.$<-env/bin/pip install -r $@
	.$<-env/bin/pip install -r $<
	echo "# You should not edit this file directly.  Instead, you should edit one of the following files ($^) and run make $@" >| $@
	.$<-env/bin/pip freeze >> $@
	cp -f $@ starfish/REQUIREMENTS-STRICT.txt
	rm -rf .$<-env

help-requirements:
	$(call print_help, refresh_all_requirements, regenerate requirements files)
	$(call print_help, check_requirements, fail if requirements files have been modified)

.PHONY: refresh_all_requirements starfish/REQUIREMENTS-STRICT.txt
#
##############################################################

### INTEGRATION ##############################################
#
include notebooks/subdir.mk

slow: fast run_notebooks docker

docker:
	docker build -t spacetx/starfish .
	docker run -ti --rm spacetx/starfish build --fov-count 1 --primary-image-dimensions '{"z": 1}' /tmp/

help-integration:
	$(call print_help, slow, alias for 'fast run_notebooks docker')
	$(call print_help, run_notebooks, run all files matching 'notebooks/py/*.py')
	$(call print_help, docker, build docker and run a simple container)

.PHONY: slow docker
#
##############################################################

### INSTALL ##################################################
#
install-dev:
	pip install --force-reinstall --upgrade -r REQUIREMENTS-STRICT.txt
	pip install -e .
	pip install -r REQUIREMENTS-CI.txt
	pip freeze

install-src:
	pip install --force-reinstall --upgrade -e .
	pip freeze

install-released-notebooks-support:
	pip install starfish
	pip install -r REQUIREMENTS-CI.txt
	pip freeze

help-install:
	$(call print_help, install-dev, pip install from the current directory with pinned requirements and tooling for CI)
	$(call print_help, install-src, pip install from the current directory)
	$(call print_help, install-released-notebooks, pip install tooling to run notebooks against the released version of starfish)

.PHONY: install-dev install-src install-released-notebooks-support help-install
#
###############################################################

help: help-main help-parts
help-main:
	@echo Main starfish make targets:
	@echo =======================================================================================
	$(call print_help, help, print this text)
help-parts: help-unit help-docs help-requirements help-integration help-install
	@echo =======================================================================================
	@echo Default: all

.PHONY: help help-unit help-requirements help-integration help-install
