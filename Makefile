SHELL := /bin/bash

EDITOR?=vi

MPLBACKEND?=Agg
export MPLBACKEND

MODULES=starfish examples/data_formatting_examples/format*

DOCKER_IMAGE?=spacetx/starfish
DOCKER_BUILD?=1

VERSION=$(shell sh -c "git describe --exact --dirty 2> /dev/null")
# if you update this, you will need to update the version pin for the "Install Napari & Test napari (pinned)" test in .travis.yml
PIP_VERSION=21.2.4

define print_help
    @printf "    %-28s   $(2)\n" $(1)
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

# note that napari tests shouldn't be run in parallel because Qt seems to intermittently fail when multiple QApplications are spawned on threads.
test:
	pytest -v -n 8 --cov starfish -m 'not napari'
	pytest -v --cov starfish -m 'napari'

fast-test:
	pytest -v -n 8 --cov starfish -m 'not (slow or napari)'

slow-test:
	pytest -v -n 8 --cov starfish -m 'slow and (not napari)'

# note that this shouldn't be run in parallel because Qt seems to intermittently fail when multiple QApplications are spawned on threads.
napari-test:
	pytest -v --cov starfish -m 'napari'

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
GENERATED_REQUIREMENT_FILES=starfish/REQUIREMENTS-STRICT.txt requirements/REQUIREMENTS-CI.txt requirements/REQUIREMENTS-NAPARI-CI.txt
SOURCE_REQUIREMENT_FILES=REQUIREMENTS.txt requirements/REQUIREMENTS-CI.txt.in requirements/REQUIREMENTS-NAPARI-CI.txt.in

# This rule pins the requirements with the minimal set of changes required to satisfy the
# requirements.  This is typically run when a new requirement is added, and we want to
# propagate the new requirement to the pin files.
pin-requirements : $(GENERATED_REQUIREMENT_FILES)

# This rule removes all existing pins and pins all requirements based on the latest set of libraries
# that satisfy the requirements.  This is typically run periodically to make sure the pins don't
# become too stale.
pin-all-requirements:
	@for target in $(GENERATED_REQUIREMENT_FILES); do \
		echo -n '' >| $$target; \
	done
	@if [ $$(uname -s) == "Darwin" ]; then sleep 1; fi  # this is require because Darwin HFS+ only has second-resolution for timestamps.
	@touch $(SOURCE_REQUIREMENT_FILES)
	$(MAKE) $(GENERATED_REQUIREMENT_FILES)

starfish/REQUIREMENTS-STRICT.txt : REQUIREMENTS.txt
	[ ! -e .$<-env ] || exit 1
	$(call create_venv, .$<-env)
	.$<-env/bin/pip install --upgrade pip==$(PIP_VERSION)
	.$<-env/bin/pip install -r $@
	.$<-env/bin/pip install -r $<
	echo "# You should not edit this file directly.  Instead, you should edit one of the following files ($^) and run make $@" >| $@
	.$<-env/bin/pip freeze --all | grep -v "pip==$(PIP_VERSION)" >> $@
	rm -rf .$<-env

requirements/REQUIREMENTS-%.txt : requirements/REQUIREMENTS-%.txt.in REQUIREMENTS.txt
	[ ! -e .$<-env ] || exit 1
	$(call create_venv, .$<-env)
	.$<-env/bin/pip install --upgrade pip==$(PIP_VERSION)
	.$<-env/bin/pip install -r $@
	for src in $^; do \
		.$<-env/bin/pip install -r $$src; \
	done
	echo "# You should not edit this file directly.  Instead, you should edit one of the following files ($<) and run make $@" >| $@
	.$<-env/bin/pip freeze --all | grep -v "pip==$(PIP_VERSION)" >> $@
	rm -rf .$<-env

check-requirements:
	if [[ $$(git status --porcelain $(GENERATED_REQUIREMENT_FILES)) ]]; then \
	    echo "Modifications found in REQUIREMENTS files"; exit 2; \
	fi

help-requirements:
	$(call print_help, pin-requirements, pin requirements with minimal set of changes)
	$(call print_help, pin-all-requirements, pin requirements with latest packages)
	$(call print_help, check-requirements, fail if requirements files have been modified)

.PHONY: pin-requirements pin-all-requirements check-requirements
#
##############################################################

### INTEGRATION ##############################################
#
include notebooks/subdir.mk
include examples/pipelines/subdir.mk

test-examples: export TESTING=1
test-examples: run-examples
slow: fast run-notebooks test-examples docker

docker:
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	docker run -ti --rm $(DOCKER_IMAGE) starfish --help


help-integration:
	$(call print_help, slow, alias for 'fast run-notebooks docker')
	$(call print_help, run-notebooks, run all files matching 'notebooks/py/*.py')
	$(call print_help, docker, build docker and run a simple container)

.PHONY: slow docker
#
##############################################################

### INSTALL ##################################################
#
install-dev:
	python -m pip install --upgrade pip==$(PIP_VERSION)
	pip install -r requirements/REQUIREMENTS-CI.txt
	pip install -e .
	pip freeze

install-src:
	python -m pip install --upgrade pip==$(PIP_VERSION) -e .
	pip freeze

install-released-notebooks-support:
	python -m pip install --upgrade pip==$(PIP_VERSION)
	pip install -r requirements/REQUIREMENTS-CI.txt
	pip install starfish
	pip freeze

help-install:
	$(call print_help, install-dev, pip install from the current directory with pinned requirements and tooling for CI)
	$(call print_help, install-src, pip install from the current directory)
	$(call print_help, install-released-notebooks, pip install tooling to run notebooks against the released version of starfish)

.PHONY: install-dev install-src install-released-notebooks-support help-install
#
###############################################################

### Deployment ################################################
#
# General release steps:
# --------------------------------------------------------------
#
#  (0) Pull master and check out the latest version of master for
#      releasing.
#
#  (1) `make release-changelog` to print a suggested update to
#      CHANGELOG.md. Replace "XXX" with your intended tag and
#      perform other edits. Important is that each section is
#      separated by a line beginning with '##'.
#
#      Remove changelog entries that do not pertain to end users.
#
#  (2) Commit all files and remove any untracked files.
#      `git status` should show nothing.
#
#  (3) Push master to origin, i.e., `git push origin master`.
#
#  (4) Run `make release-tag TAG=x.y.z` to tag your release.
#
#  (5) Run `make release-prep` which:
#     - checks the tag
#     - creates a virtualenv
#     - builds and installs the sdist
#
#  (6) Run `make release-verify` which:
#     - runs tests
#     - builds docker
#
#  (7) Run `make release-upload` and execute the commands.
#
#  If anything goes wrong, rollback the various steps:
#     - delete on docker hub
#     - delete local docker image
#     - delete tag locally
#     - make clean

## Sections: 1 - 3

# public: print a changelog to stdout
release-changelog:
	@if test -n "$(VERSION)"; then                    \
		echo VERSION is set to $(VERSION)         \
		echo Create your changelog before tagging.\
		exit 102;                                 \
	fi;
	@echo "##" "[XXX]" - $(shell sh -c "date +'%Y-%m-%d'")
	@git log $(shell sh -c "git describe --tags --abbrev=0")..HEAD --pretty=format:"- %s"
	@printf "\n\n"
	@cat CHANGELOG.md; echo "[XXX]: https://github.com/spacetx/starfish/releases/tag/XXX"

# public: generate a tag from the current commit & changelog
release-tag:
	@if test -z "$(TAG)"; then                     \
		echo TAG is not set. Use:              \
		echo make TAG=x.y.z release-tag;       \
		exit 104;                              \
	fi &&                                          \
	printf "Tag $(TAG)\n\n" > release-msg &&       \
	sed -n -e '/^##/,/^##/{ /^##/d; /^##/d; p; }' CHANGELOG.md >> release-msg && \
	$(EDITOR) release-msg &&                      \
	git tag -a -F release-msg "$(TAG)" &&          \
	rm release-msg

## Sections: 4

# private: assert a clean tag on the current commit
release-check:
	@if test -z "$(VERSION)"; then                    \
		echo VERSION is not set\!;                \
		echo Is the current commit tagged?;       \
		echo If not, create a tag for the current version.;\
		exit 100;                                 \
	elif [[ "$(VERSION)" == *"dirty"* ]] ; then       \
		echo VERSION is dirty.;                   \
		echo Please commit all files and re-tag.; \
		exit 101;                                 \
	else                                              \
		echo "===============================";   \
		echo "Releasing version: $(VERSION)";     \
		echo "===============================";   \
	fi;

# private: assert release-env and release-msg don't exist
release-ready:
	@if compgen -G release-* > /dev/null; then        \
		echo "Previous release found.";           \
		echo "Run 'make clean'";                  \
		exit 103;                                 \
	fi;

# private: create a virtualenv for testing the release
release-env: release-env/bin/activate release-env/bin/make_shell

# private: call virtualenv and pip install
release-env/bin/activate:
	$(call create_venv, release-env)
	release-env/bin/pip install --force-reinstall --upgrade pip==$(PIP_VERSION)
	release-env/bin/pip install -r requirements/REQUIREMENTS-CI.txt
	touch release-env/bin/activate

# private: create make_shell for activating the virtualenv below
release-env/bin/make_shell:
	echo '#!/bin/bash' > $@
	echo 'source release-env/bin/activate' >> $@
	echo 'bash "$$@"' >> $@
	chmod a+x $@
# public: generate the release build
release-prep: release-check release-ready release-env
	release-env/bin/python setup.py clean
	release-env/bin/python setup.py sdist
	release-env/bin/pip install dist/starfish-$(VERSION).tar.gz

## Sections: 5 - 6

# public: run tests on the current release build
release-verify: export SHELL=release-env/bin/make_shell
release-verify: release-check slow release-docker

# public: tag the docker images
release-docker: release-check
	docker tag $(DOCKER_IMAGE) $(DOCKER_IMAGE):$(VERSION)
	docker tag $(DOCKER_IMAGE) $(DOCKER_IMAGE):$(VERSION)-$(DOCKER_BUILD)

# public: print commands for uploading artifacts
release-upload: release-check
	@printf '\n# Please execute the following steps\n'
	@echo git push origin $(VERSION)
	@echo docker push $(DOCKER_IMAGE)
	@echo docker push $(DOCKER_IMAGE):$(VERSION)
	@echo docker push $(DOCKER_IMAGE):$(VERSION)-$(DOCKER_BUILD)
	@echo twine upload dist/starfish-$(VERSION).tar.gz

clean:
	rm -rf release-env
	rm -rf release-msg
	rm -rf starfish.egg-info
	rm -rf dist
	rm -rf build
	rm -rf .eggs
	rm -f .cover*

help-deployment:
	$(call print_help, release-changelog, Print changelog for updating CHANGELOG.md)
	$(call print_help, release-tag, Tag current commit with update changelog)
	$(call print_help, release-prep, Builds and installs the current tagged version)
	$(call print_help, release-verify, Runs tests on the tagged version)
	$(call print_help, release-upload, Prints commands for uploading release artifacts)
	$(call print_help, clean, Deletes build-related directories)

.PHONY: clean release-prep release-verify release-upload release-check
#
###############################################################
help: help-main help-parts
help-main:
	@echo Main starfish make targets:
	@echo =======================================================================================
	$(call print_help, help, print this text)
help-parts: help-unit help-docs help-requirements help-integration help-install help-deployment
	@echo =======================================================================================
	@echo Default: all

.PHONY: help help-unit help-requirements help-integration help-install help-release
