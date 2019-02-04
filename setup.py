#!/usr/bin/env python

import os
import setuptools

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

setuptools.setup(
    name="starfish",
    version="0.0.31",
    description="Pipelines and pipeline components for the analysis of image-based transcriptomics data",
    author="Deep Ganguli",
    author_email="dganguli@chanzuckerberg.com",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=0.15.1',
        'scipy>=1.1.0',
        'pandas>=0.23.4',
        'xarray>=0.10.8',
        'scikit-image>=0.14.0',
        'scikit-learn>=0.19.2',
        'semantic-version',
        'trackpy',
        'showit',
        'slicedimage',
        'tqdm',
        'sympy',
        'jsonschema',
        'click',
        'regional',
    ],
    extras_require={
        'napari': ['napari-gui']
    },
    entry_points={
        'console_scripts': [
            "starfish=starfish.starfish:starfish",
        ]
    },
    classifiers=CLASSIFIERS,
    include_package_data=True,
)
