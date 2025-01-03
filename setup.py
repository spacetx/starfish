#!/usr/bin/env python

from pathlib import Path

import setuptools
import versioneer


install_requires = [
    line.rstrip() for line in open(Path(__file__).parent / "REQUIREMENTS.txt")
]

with open(Path("starfish") / "core" / "_display.py") as f:
    for line in f.readlines():
        if line.startswith("NAPARI_VERSION"):
            napari_version = line.split('"')[1]
            break

setuptools.setup(
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        'napari': [f"napari[all]>={napari_version}"],
    },
    include_package_data=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
