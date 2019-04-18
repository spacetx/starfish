# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    ?= Starfish
SOURCEDIR     ?= source
BUILDDIR      ?= build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/usage/data_formatting_examples/*
	rm -rf $(SOURCEDIR)/usage/data_processing_examples/*
	rm -rf $(SOURCEDIR)/creating_an_image_processing_pipeline/tutorials/*

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
