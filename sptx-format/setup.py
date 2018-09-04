import os

from setuptools import setup

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

install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), "REQUIREMENTS.txt"))
]

setup(
    name="validate_sptx",
    version="1.1.0",
    description="test suite for validating json _schema for the spaceTx image format",
    url="https://github.com/spacetx/sptx-format.git",
    author="Ambrose J. Carr",
    author_email="mail@ambrosejcarr.com",
    packages=["validate_sptx"],
    install_requires=install_requires,
    entry_points={
        'console_scripts': "validate-sptx=validate_sptx.validate_sptx:validate_sptx"
    },
    include_package_data=True,
    classifiers=CLASSIFIERS,
)
