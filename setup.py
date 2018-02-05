#!/usr/bin/env python

import os
import setuptools

install_requires = [line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), "REQUIREMENTS.txt"))]

setuptools.setup(
    name="starfish",
    version="0.0.2",
    description="Pipelines and pipeline components for the analysis of image-based transcriptomics data",
    author="Deep Ganguli",
    author_email="dganguli@chanzuckerberg.com",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': "starfish=starfish.starfish:starfish"
    }
)
